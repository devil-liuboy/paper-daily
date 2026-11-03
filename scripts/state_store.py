"""Persist non-secret Paper Daily state on a dedicated GitHub branch.

Budget updates use the SHA saved at hydrate time, never a freshly fetched SHA.
That compare-and-swap rule prevents an older workflow from erasing newer paid
attempts. The paper cache has separate files and cannot reset the ledger.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any


BRANCH = "paper-state"
MAX_FILE_BYTES = 10 * 1024 * 1024
_SHA = re.compile(r"^[0-9a-f]{40}$")
_PAPER_FILES = ("papers.json", "conference_papers.json")


class StateStoreError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _settings() -> tuple[str, str]:
    repository = os.getenv("GITHUB_REPOSITORY", "")
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise StateStoreError("A valid GITHUB_REPOSITORY owner/repository is required.")
    if not token:
        raise StateStoreError("GITHUB_TOKEN is required for persistent state.")
    return repository, token


def _request(method: str, route: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    _, token = _settings()
    if not route.startswith("/repos/") or "://" in route or "#" in route:
        raise StateStoreError("Invalid state API route.")
    request = urllib.request.Request(
        "https://api.github.com" + route,
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        method=method,
        headers={
            "Authorization": "Bearer " + token,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "personal-paper-daily-state/1.0",
        },
    )
    try:
        with urllib.request.build_opener(_NoRedirect()).open(request, timeout=60) as response:
            raw = response.read(2 * MAX_FILE_BYTES + 1)
        if len(raw) > 2 * MAX_FILE_BYTES:
            raise ValueError
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError
        return data
    except urllib.error.HTTPError as exc:
        # Do not echo URL, response body, headers, or provider diagnostics.
        raise StateStoreError(f"GitHub state API returned HTTP {exc.code}; state was not reset.", exc.code) from None
    except StateStoreError:
        raise
    except Exception:
        raise StateStoreError("GitHub state API transport or response failure; state was not reset.") from None


def _prefix() -> str:
    repository, _ = _settings()
    return "/repos/" + repository


def _file_route(name: str) -> str:
    if name not in {"usage.json", *_PAPER_FILES}:
        raise StateStoreError("Unsupported state filename.")
    return _prefix() + "/contents/" + urllib.parse.quote(name)


def _valid_sha(value: Any) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise StateStoreError("GitHub state response has an invalid SHA.")
    return value


def _get_file(name: str) -> tuple[bytes, str] | None:
    try:
        metadata = _request("GET", _file_route(name) + "?ref=" + BRANCH)
    except StateStoreError as exc:
        if exc.status == 404:
            return None
        raise
    sha = _valid_sha(metadata.get("sha"))
    if metadata.get("type", "file") != "file":
        raise StateStoreError("State path is not a regular file.")
    size = metadata.get("size", 0)
    if type(size) is not int or size < 0 or size > MAX_FILE_BYTES:
        raise StateStoreError("State file exceeds the supported size.")
    if metadata.get("encoding") == "none":
        # Contents API omits inline data above 1 MiB. Fetch the exact blob SHA,
        # not a moving branch URL, to keep the content and revision consistent.
        metadata = _request("GET", _prefix() + "/git/blobs/" + sha)
    if metadata.get("encoding") != "base64" or not isinstance(metadata.get("content"), str):
        raise StateStoreError("State file has an unsupported content encoding.")
    try:
        raw = base64.b64decode("".join(metadata["content"].split()), validate=True)
    except (ValueError, binascii.Error):
        raise StateStoreError("State file contains invalid encoded data.") from None
    if len(raw) > MAX_FILE_BYTES:
        raise StateStoreError("State file exceeds the supported size.")
    return raw, sha


def _put_file(name: str, raw: bytes, sha: str | None, message: str) -> str:
    if len(raw) > MAX_FILE_BYTES:
        raise StateStoreError("State file exceeds the supported size.")
    body: dict[str, Any] = {"message": message, "content": base64.b64encode(raw).decode("ascii"), "branch": BRANCH}
    if sha is not None:
        body["sha"] = _valid_sha(sha)
    result = _request("PUT", _file_route(name), body)
    content = result.get("content")
    if not isinstance(content, dict):
        raise StateStoreError("GitHub did not confirm the state file revision.")
    return _valid_sha(content.get("sha"))


def _allow_init() -> bool:
    return os.getenv("BUDGET_ALLOW_INIT", "").lower() == "true"


def _ensure_branch() -> None:
    try:
        _request("GET", _prefix() + "/git/ref/heads/" + BRANCH)
        return
    except StateStoreError as exc:
        if exc.status != 404:
            raise
    if not _allow_init():
        raise StateStoreError("State branch missing; restore it or explicitly authorize initialization.")
    main = _request("GET", _prefix() + "/git/ref/heads/main")
    sha = _valid_sha(main.get("object", {}).get("sha"))
    _request("POST", _prefix() + "/git/refs", {"ref": "refs/heads/" + BRANCH, "sha": sha})


def _budget_path() -> Path:
    return Path(os.getenv("LLM_BUDGET_PATH", ".paper-state/usage.json"))


def _revision_path(path: Path) -> Path:
    return path.parent / "revision.json"


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError:
        raise StateStoreError("Unable to save local state atomically.") from None
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _encode(data: dict[str, Any]) -> bytes:
    return (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _save_revision(path: Path, sha: str) -> None:
    repository, _ = _settings()
    _atomic_write(_revision_path(path), _encode({"version": 1, "repository": repository, "branch": BRANCH, "usage_sha": sha}))


def _load_revision(path: Path) -> str:
    repository, _ = _settings()
    try:
        revision = json.loads(_revision_path(path).read_text(encoding="utf-8"))
        if revision.get("version") != 1 or revision.get("repository") != repository or revision.get("branch") != BRANCH:
            raise ValueError
        return _valid_sha(revision.get("usage_sha"))
    except (OSError, ValueError, AttributeError, StateStoreError):
        raise StateStoreError("Budget revision is missing or invalid; hydrate before making paid calls.") from None


def _read_valid_budget(path: Path) -> dict[str, Any]:
    try:
        try:
            from scripts.llm_budget import _read
        except ModuleNotFoundError:
            from llm_budget import _read
        return _read(path, allow_init=False)
    except Exception:
        raise StateStoreError("Budget ledger is invalid; refusing to replace persistent spending history.") from None


def _install_budget(path: Path, raw: bytes) -> None:
    # Validate a temporary copy first, so corrupt remote data cannot replace
    # the local ledger or make a partly written ledger look initialized.
    candidate = path.with_name(path.name + ".validate-" + uuid.uuid4().hex)
    try:
        _atomic_write(candidate, raw)
        _read_valid_budget(candidate)
        os.replace(candidate, path)
    except OSError:
        raise StateStoreError("Unable to install the restored budget ledger.") from None
    finally:
        try:
            candidate.unlink(missing_ok=True)
        except OSError:
            pass


def hydrate() -> None:
    """Restore remote ledger; initialization is explicit and never a reset."""
    _ensure_branch()
    remote = _get_file("usage.json")
    if remote is None:
        if not _allow_init():
            raise StateStoreError("Remote budget ledger missing; refusing an implicit spending reset.")
        # PUT without a SHA is create-only. A race fails rather than replacing
        # a ledger another workflow may have just initialized.
        _put_file("usage.json", _encode({"version": 1, "months": {}}), None, "Initialize paper budget ledger")
        remote = _get_file("usage.json")
        if remote is None:
            raise StateStoreError("The initialized budget ledger could not be verified.")
    raw, sha = remote
    path = _budget_path()
    _install_budget(path, raw)
    _save_revision(path, sha)
    restore()


def persist_budget(path: Path | str | None = None) -> None:
    """CAS-write the ledger using only the revision from the last hydrate/PUT."""
    path = Path(path) if path is not None else _budget_path()
    sha = _load_revision(path)
    ledger = _read_valid_budget(path)
    new_sha = _put_file("usage.json", _encode(ledger), sha, "Record paper summary budget usage")
    _save_revision(path, new_sha)


def _validate_papers(raw: bytes) -> None:
    try:
        data = json.loads(raw)
        if not isinstance(data, dict) or not isinstance(data.get("papers"), list):
            raise ValueError
    except (ValueError, TypeError):
        raise StateStoreError("Paper cache is invalid; refusing to replace it.") from None


def snapshot() -> None:
    """Persist daily/conference data separately; never touches usage.json."""
    for name in _PAPER_FILES:
        path = Path("web/data") / name
        try:
            raw = path.read_bytes()
        except OSError:
            raise StateStoreError("A generated paper cache file is missing or unreadable.") from None
        _validate_papers(raw)
        current = _get_file(name)
        sha = current[1] if current is not None else None
        _put_file(name, raw, sha, "Save collected paper cache: " + name)


def restore() -> None:
    """Restore available paper files; missing first-run files are allowed."""
    for name in _PAPER_FILES:
        remote = _get_file(name)
        if remote is None:
            continue
        raw, _ = remote
        _validate_papers(raw)
        _atomic_write(Path("web/data") / name, raw)


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage persistent Paper Daily state.")
    parser.add_argument("command", choices=("hydrate", "persist", "snapshot", "restore"))
    args = parser.parse_args()
    try:
        {"hydrate": hydrate, "persist": persist_budget, "snapshot": snapshot, "restore": restore}[args.command]()
    except StateStoreError as exc:
        print("State store failed: " + str(exc), file=sys.stderr)
        return 1
    print("State " + args.command + " complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
