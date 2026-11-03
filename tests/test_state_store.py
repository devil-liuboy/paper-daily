import base64
import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from scripts import state_store as state


SHA1, SHA2, SHA3 = "a" * 40, "b" * 40, "c" * 40
EMPTY = {"version": 1, "months": {}}


def existing_ledger():
    return {"version": 1, "months": {"2026-11": [{
        "id": "one", "day": "2026-11-03", "status": "reserved",
        "reserved_micros": 30000, "charged_micros": 30000,
    }]}}


class StateStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix=".test-state-", dir=Path.cwd())
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / ".paper-state" / "usage.json"
        self.env = mock.patch.dict(os.environ, {
            "GITHUB_REPOSITORY": "example/paper-daily", "GITHUB_TOKEN": "test-not-real-token",
            "LLM_BUDGET_PATH": str(self.path),
        }, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def seed_local(self, ledger=EMPTY, sha=SHA1):
        state._atomic_write(self.path, state._encode(ledger))
        state._save_revision(self.path, sha)

    def test_hydrate_restores_existing_even_when_initialization_allowed(self):
        os.environ["BUDGET_ALLOW_INIT"] = "true"
        self.seed_local()
        with mock.patch.object(state, "_ensure_branch"), mock.patch.object(state, "_get_file", return_value=(state._encode(existing_ledger()), SHA2)), mock.patch.object(state, "restore"):
            with mock.patch.object(state, "_put_file") as put:
                state.hydrate()
                put.assert_not_called()
        self.assertEqual(json.loads(self.path.read_text()), existing_ledger())
        self.assertEqual(state._load_revision(self.path), SHA2)

    def test_missing_ledger_fails_closed_without_explicit_init(self):
        with mock.patch.object(state, "_ensure_branch"), mock.patch.object(state, "_get_file", return_value=None):
            with self.assertRaises(state.StateStoreError):
                state.hydrate()
        self.assertFalse(self.path.exists())

    def test_initialization_creates_then_reads_back_remote(self):
        os.environ["BUDGET_ALLOW_INIT"] = "true"
        with mock.patch.object(state, "_ensure_branch"), mock.patch.object(state, "restore"):
            with mock.patch.object(state, "_get_file", side_effect=[None, (state._encode(EMPTY), SHA1)]) as get:
                with mock.patch.object(state, "_put_file", return_value=SHA1) as put:
                    state.hydrate()
        self.assertEqual(get.call_count, 2)
        self.assertIsNone(put.call_args.args[2])
        self.assertEqual(state._load_revision(self.path), SHA1)

    def test_persist_uses_saved_sha_without_fetching_a_new_sha(self):
        self.seed_local(existing_ledger())
        with mock.patch.object(state, "_get_file") as get, mock.patch.object(state, "_put_file", return_value=SHA2) as put:
            state.persist_budget(self.path)
        get.assert_not_called()
        self.assertEqual(put.call_args.args[2], SHA1)
        self.assertEqual(state._load_revision(self.path), SHA2)

    def test_cas_conflict_does_not_overwrite_or_update_revision(self):
        self.seed_local(existing_ledger())
        with mock.patch.object(state, "_put_file", side_effect=state.StateStoreError("conflict", 409)) as put:
            with self.assertRaises(state.StateStoreError):
                state.persist_budget(self.path)
        self.assertEqual(put.call_count, 1)
        self.assertEqual(state._load_revision(self.path), SHA1)

    def test_persist_requires_hydrated_revision(self):
        state._atomic_write(self.path, state._encode(EMPTY))
        with mock.patch.object(state, "_put_file") as put:
            with self.assertRaises(state.StateStoreError):
                state.persist_budget(self.path)
        put.assert_not_called()

    def test_corrupt_remote_does_not_replace_local_budget(self):
        self.seed_local(existing_ledger())
        with mock.patch.object(state, "_ensure_branch"), mock.patch.object(state, "_get_file", return_value=(b"{}", SHA2)):
            with self.assertRaises(state.StateStoreError):
                state.hydrate()
        self.assertEqual(json.loads(self.path.read_text()), existing_ledger())
        self.assertEqual(state._load_revision(self.path), SHA1)

    def test_new_branch_requires_authorized_init_and_uses_main_head(self):
        with mock.patch.object(state, "_request", side_effect=state.StateStoreError("not found", 404)) as request:
            with self.assertRaises(state.StateStoreError):
                state._ensure_branch()
            self.assertEqual(request.call_count, 1)
        os.environ["BUDGET_ALLOW_INIT"] = "true"
        with mock.patch.object(state, "_request", side_effect=[state.StateStoreError("not found", 404), {"object": {"sha": SHA3}}, {}]) as request:
            state._ensure_branch()
        self.assertEqual(request.call_args.args[0], "POST")
        self.assertEqual(request.call_args.args[2], {"ref": "refs/heads/paper-state", "sha": SHA3})

    def test_api_request_uses_only_official_host_and_hides_error_body(self):
        opener = mock.Mock()
        opener.open.side_effect = urllib.error.HTTPError("https://attacker.invalid/token", 403, "SECRET BODY", {}, io.BytesIO(b"SECRET BODY"))
        with mock.patch.object(state.urllib.request, "build_opener", return_value=opener):
            with self.assertRaises(state.StateStoreError) as error:
                state._request("GET", "/repos/example/paper-daily/git/ref/heads/main")
        request = opener.open.call_args.args[0]
        self.assertTrue(request.full_url.startswith("https://api.github.com/"))
        self.assertNotIn("SECRET", str(error.exception))
        self.assertNotIn("token", str(error.exception))
        self.assertIsNone(state._NoRedirect().redirect_request(None, None, 302, "", {}, "https://attacker.invalid"))

    def test_large_file_uses_exact_blob_sha(self):
        raw = b'{"papers": []}'
        with mock.patch.object(state, "_request", side_effect=[
            {"sha": SHA1, "size": 2_000_000, "encoding": "none", "content": ""},
            {"encoding": "base64", "content": base64.b64encode(raw).decode()},
        ]) as request:
            self.assertEqual(state._get_file("papers.json"), (raw, SHA1))
        self.assertTrue(request.call_args.args[1].endswith("/git/blobs/" + SHA1))

    def test_snapshot_only_writes_paper_files(self):
        with mock.patch.object(Path, "read_bytes", return_value=b'{"papers": []}'):
            with mock.patch.object(state, "_get_file", return_value=None), mock.patch.object(state, "_put_file", return_value=SHA1) as put:
                state.snapshot()
        self.assertEqual([call.args[0] for call in put.call_args_list], ["papers.json", "conference_papers.json"])

    def test_restore_skips_missing_first_run_files(self):
        with mock.patch.object(state, "_get_file", return_value=None), mock.patch.object(state, "_atomic_write") as write:
            state.restore()
        write.assert_not_called()

    def test_wrong_repository_revision_is_rejected(self):
        self.seed_local()
        os.environ["GITHUB_REPOSITORY"] = "another/repository"
        with self.assertRaises(state.StateStoreError):
            state.persist_budget(self.path)


if __name__ == "__main__":
    unittest.main()
