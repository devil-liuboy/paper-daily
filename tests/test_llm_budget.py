import concurrent.futures
import datetime as dt
import io
import json
import os
import tempfile
import types
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from scripts import llm_budget as budget


def response(content=None, prompt_tokens=1000, completion_tokens=500, finish_reason="stop"):
    return io.BytesIO(json.dumps({
        "choices": [{"finish_reason": finish_reason, "message": {"content": content if content is not None else '{"problem": "test"}'}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }).encode())


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix=".test-budget-", dir=Path.cwd())
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "usage.json"
        self.env = mock.patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "test-key-never-real", "LLM_MODEL": "deepseek-flash",
            "DEEPSEEK_INPUT_CNY_PER_MILLION": "2", "DEEPSEEK_OUTPUT_CNY_PER_MILLION": "8",
            "LLM_BUDGET_PATH": str(self.path), "BUDGET_ALLOW_INIT": "true",
        }, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def records(self):
        return next(iter(json.loads(self.path.read_text())["months"].values()))

    def opener(self, result=None):
        opener = mock.Mock()
        opener.open.return_value = result if result is not None else response()
        return opener

    def test_reservation_precedes_network_and_is_settled_conservatively(self):
        opener = self.opener()
        def open_call(request, timeout):
            record = self.records()[0]
            self.assertEqual(record["status"], "reserved")
            self.assertGreater(record["reserved_micros"], 6000)
            payload = json.loads(request.data)
            self.assertEqual(request.full_url, budget.ENDPOINT)
            self.assertEqual(payload["model"], "deepseek-flash")
            self.assertEqual(payload["thinking"], {"type": "disabled"})
            self.assertEqual(payload["max_tokens"], 1800)
            return response()
        opener.open.side_effect = open_call
        with mock.patch.object(budget.urllib.request, "build_opener", return_value=opener):
            self.assertEqual(budget.call_deepseek("test paper"), {"problem": "test"})
        record = self.records()[0]
        self.assertEqual(record["charged_micros"], 6000)
        self.assertEqual(record["status"], "completed")
        ledger = self.path.read_text()
        self.assertNotIn("test-key", ledger)
        self.assertNotIn("test paper", ledger)

    def test_missing_ledger_does_not_reset_budget(self):
        os.environ.pop("BUDGET_ALLOW_INIT")
        with mock.patch.object(budget.urllib.request, "build_opener") as opener:
            with self.assertRaises(budget.BudgetError):
                budget.call_deepseek("paper")
            opener.assert_not_called()

    def test_invalid_ledger_not_reinitialized(self):
        self.path.write_text("{}")
        with self.assertRaises(budget.BudgetError):
            budget.call_deepseek("paper")
        self.assertEqual(self.path.read_text(), "{}")

    def test_no_openai_fallback_or_custom_endpoint(self):
        os.environ.pop("DEEPSEEK_API_KEY")
        os.environ["OPENAI_API_KEY"] = "do-not-use"
        with self.assertRaises(budget.BudgetError):
            budget.call_deepseek("paper")
        os.environ["DEEPSEEK_API_KEY"] = "test"
        os.environ["LLM_BASE_URL"] = "https://example.invalid"
        with self.assertRaises(budget.BudgetError):
            budget.call_deepseek("paper")

    def test_http_failure_consumes_attempt_and_full_reservation(self):
        opener = self.opener()
        opener.open.side_effect = urllib.error.HTTPError("https://api.deepseek.com", 429, "secret-error", {}, None)
        with mock.patch.object(budget.urllib.request, "build_opener", return_value=opener):
            with self.assertRaises(budget.DeepSeekError) as error:
                budget.call_deepseek("paper")
        self.assertNotIn("secret-error", str(error.exception))
        record = self.records()[0]
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["charged_micros"], record["reserved_micros"])
        opener.open.assert_called_once()

    def test_invalid_json_and_truncation_retain_reservation(self):
        for content, finish in [("not-json", "stop"), ('{"ok":true}', "length")]:
            with mock.patch.object(budget.urllib.request, "build_opener", return_value=self.opener(response(content, finish_reason=finish))):
                with self.assertRaises(budget.DeepSeekError):
                    budget.call_deepseek("paper")
        self.assertTrue(all(record["status"] == "failed" for record in self.records()))

    def test_daily_limit_survives_multiple_calls(self):
        os.environ["LLM_DAILY_ATTEMPT_LIMIT"] = "1"
        with mock.patch.object(budget.urllib.request, "build_opener", return_value=self.opener()):
            budget.call_deepseek("paper")
            with self.assertRaises(budget.BudgetError):
                budget.call_deepseek("another paper")
        self.assertEqual(len(self.records()), 1)

    def test_reservation_blocks_overspending_before_network(self):
        os.environ["LLM_MONTHLY_BUDGET_CNY"] = "0.001"
        with mock.patch.object(budget.urllib.request, "build_opener") as opener:
            with self.assertRaises(budget.BudgetError):
                budget.call_deepseek("paper")
            opener.assert_not_called()

    def test_concurrent_reservations_respect_daily_limit(self):
        os.environ["LLM_DAILY_ATTEMPT_LIMIT"] = "2"
        opener = self.opener()
        opener.open.side_effect = lambda *args, **kwargs: response()
        def call():
            try:
                return budget.call_deepseek("paper")
            except budget.BudgetError:
                return None
        with mock.patch.object(budget.urllib.request, "build_opener", return_value=opener):
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(lambda _: call(), range(8)))
        self.assertEqual(sum(item is not None for item in results), 2)
        self.assertEqual(len(self.records()), 2)

    def test_unconfigured_price_and_oversize_prompt_are_free_failures(self):
        os.environ.pop("DEEPSEEK_INPUT_CNY_PER_MILLION")
        with self.assertRaises(budget.BudgetError):
            budget.call_deepseek("paper")
        os.environ["DEEPSEEK_INPUT_CNY_PER_MILLION"] = "2"
        with self.assertRaises(budget.BudgetError):
            budget.call_deepseek("中" * 6000)
        self.assertFalse(self.path.exists())

    def test_redirect_is_not_followed(self):
        self.assertIsNone(budget._NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.invalid"))

    def test_stats_are_public_safe_and_reflect_retained_reservations(self):
        with mock.patch.object(budget.urllib.request, "build_opener", return_value=self.opener()):
            budget.call_deepseek("private prompt")
        stats = budget.get_budget_stats()
        self.assertEqual(stats["daily_attempts"], 1)
        self.assertEqual(stats["accounted_cny"], 0.006)
        self.assertEqual(stats["remaining_cny"], 19.994)
        self.assertNotIn("private", json.dumps(stats))

    def test_remote_reservation_is_persisted_before_request(self):
        os.environ["BUDGET_PERSIST_GITHUB"] = "true"
        events = []
        persist = lambda path: events.append(self.records()[0]["status"])
        opener = self.opener()
        opener.open.side_effect = lambda *args, **kwargs: (events.append("network"), response())[1]
        with mock.patch.dict("sys.modules", {"scripts.state_store": types.SimpleNamespace(persist_budget=persist)}):
            with mock.patch.object(budget.urllib.request, "build_opener", return_value=opener):
                budget.call_deepseek("paper")
        self.assertEqual(events, ["reserved", "network", "completed"])

    def test_remote_persistence_failure_prevents_request(self):
        os.environ["BUDGET_PERSIST_GITHUB"] = "true"
        persist = mock.Mock(side_effect=RuntimeError("secret internal error"))
        with mock.patch.dict("sys.modules", {"scripts.state_store": types.SimpleNamespace(persist_budget=persist)}):
            with mock.patch.object(budget.urllib.request, "build_opener") as opener:
                with self.assertRaises(budget.BudgetError) as error:
                    budget.call_deepseek("paper")
                opener.assert_not_called()
        self.assertNotIn("secret", str(error.exception))
        self.assertEqual(self.records()[0]["status"], "reserved")

    def test_calendar_day_uses_beijing_and_keeps_prior_month(self):
        os.environ["LLM_DAILY_ATTEMPT_LIMIT"] = "1"
        opener = self.opener()
        opener.open.side_effect = lambda *args, **kwargs: response()
        with mock.patch.object(budget.urllib.request, "build_opener", return_value=opener):
            for instant in [dt.datetime(2026, 11, 30, 23, 59, tzinfo=budget.TIMEZONE), dt.datetime(2026, 12, 1, 0, 1, tzinfo=budget.TIMEZONE)]:
                with mock.patch.object(budget, "_now", return_value=instant):
                    budget.call_deepseek("paper")
        ledger = json.loads(self.path.read_text())
        self.assertEqual(set(ledger["months"]), {"2026-11", "2026-12"})
        self.assertEqual(ledger["months"]["2026-12"][0]["day"], "2026-12-01")

    def test_excessive_usage_records_cost_and_halts_later_months(self):
        opener = self.opener(response(prompt_tokens=100_000, completion_tokens=2_000))
        with mock.patch.object(budget.urllib.request, "build_opener", return_value=opener):
            with self.assertRaises(budget.DeepSeekError):
                budget.call_deepseek("paper")
        ledger = budget._read(self.path)
        record = self.records()[0]
        self.assertEqual(record["charged_micros"], 216_000)
        self.assertEqual(record["outcome"], "usage_exceeded_bound")
        self.assertEqual(ledger["halt"]["record_id"], record["id"])
        self.assertEqual(budget.get_budget_stats()["budget_status"], "halted")
        with mock.patch.object(budget, "_now", return_value=dt.datetime(2027, 1, 1, tzinfo=budget.TIMEZONE)):
            with mock.patch.object(budget.urllib.request, "build_opener") as network:
                with self.assertRaises(budget.BudgetError):
                    budget.call_deepseek("another paper")
                network.assert_not_called()
            self.assertEqual(budget.get_budget_stats()["budget_status"], "halted")
        self.assertEqual(len(self.records()), 1)

    def test_excessive_usage_in_invalid_response_also_halts(self):
        opener = self.opener(response("not JSON", prompt_tokens=100_000))
        with mock.patch.object(budget.urllib.request, "build_opener", return_value=opener):
            with self.assertRaises(budget.DeepSeekError):
                budget.call_deepseek("paper")
        self.assertEqual(self.records()[0]["charged_micros"], 204_000)
        self.assertEqual(budget.get_budget_stats()["budget_status"], "halted")

    def test_legacy_bound_violation_without_halt_flag_remains_halted(self):
        opener = self.opener(response(prompt_tokens=100_000))
        with mock.patch.object(budget.urllib.request, "build_opener", return_value=opener):
            with self.assertRaises(budget.DeepSeekError):
                budget.call_deepseek("paper")
        ledger = json.loads(self.path.read_text())
        del ledger["halt"]
        record = next(iter(ledger["months"].values()))[0]
        record["charged_micros"] = record["reserved_micros"]
        self.path.write_text(json.dumps(ledger))
        with mock.patch.object(budget.urllib.request, "build_opener") as network:
            with self.assertRaises(budget.BudgetError):
                budget.call_deepseek("paper")
            network.assert_not_called()
        self.assertEqual(budget.get_budget_stats()["budget_status"], "halted")


if __name__ == "__main__":
    unittest.main()
