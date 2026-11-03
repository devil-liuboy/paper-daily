"""Offline regression checks for the personal deployment and its trust boundaries."""
import copy
import datetime as dt
import json
import io
import os
from pathlib import Path
import unittest
from unittest import mock

from scripts import collect_papers as c


class PersonalCollectorTest(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.topic = c.Topic("gui", "GUI Agent", "GUI agent efficiency", ["GUI agent"], ["cs.AI"], 2.0)
        self.config = {
            "sources": [{"type": "arxiv", "name": "arXiv"}],
            "conference_sources": {"enabled": False},
            "topics": [self.topic.__dict__],
        }

    def paper(self, title="GUI agent efficiency"):
        now = dt.datetime.now(dt.timezone.utc)
        return {"id": "2611.00001v1", "source": "arXiv", "title": title,
                "summary": "We study GUI agent efficiency with vision language models. " * 5,
                "published": now.isoformat(), "updated": now.isoformat(),
                "authors": ["Test Author"], "categories": ["cs.AI"]}

    def run_collect(self, fetched=None, previous=None, failure=None, max_summaries=10, max_new_papers=15):
        with mock.patch.object(c, "load_json", return_value=self.config), \
             mock.patch.object(c, "load_existing_payload", side_effect=[previous or {}, {}]), \
             mock.patch.object(c, "fetch_source_topic", return_value=fetched or [], side_effect=failure), \
             mock.patch.object(c.time, "sleep"), \
             mock.patch.object(c, "write_json") as writer:
            self.last_writer = writer
            result = c.collect(Path("config"), Path("daily"), Path("conference"),
                               7, 20, max_summaries, max_new_papers, 450, 0, 0, 0, True, 45, False)
            return result, writer.call_count

    def test_issue_config_is_disabled_by_default(self):
        with mock.patch.object(c, "github_request") as req:
            self.assertIs(c.load_issue_config(self.config), self.config)
        req.assert_not_called()

    def test_untrusted_issue_cannot_set_topics_or_sources(self):
        os.environ.update(ENABLE_ISSUE_CONFIG="true", GITHUB_TOKEN="dummy", GITHUB_REPOSITORY="owner/repo")
        issue = {"title": "Research Interests", "user": {"login": "outsider"},
                 "body": json.dumps({"topics": [{"name": "untrusted"}], "sources": [{"type": "feed"}]})}
        with mock.patch.object(c, "github_request", return_value=[issue]):
            self.assertIs(c.load_issue_config(self.config), self.config)

    def test_owner_issue_can_only_override_topics(self):
        os.environ.update(ENABLE_ISSUE_CONFIG="true", GITHUB_TOKEN="dummy", GITHUB_REPOSITORY="owner/repo")
        issue = {"title": "Research Interests", "user": {"login": "owner"},
                 "body": json.dumps({"topics": [{"name": "updated"}], "sources": [{"type": "feed", "url": "https://example.invalid"}]})}
        with mock.patch.object(c, "github_request", return_value=[issue]):
            config = c.load_issue_config(self.config)
        self.assertEqual(config["topics"][0]["name"], "updated")
        self.assertEqual(config["sources"], self.config["sources"])

    def test_feed_cannot_read_model_key(self):
        source = c.SourceConfig("feed", "untrusted", "https://example.invalid", bearer_token_env="DEEPSEEK_API_KEY")
        with self.assertRaisesRegex(ValueError, "CUSTOM_FEED"):
            c.source_request_headers(source)

    def test_authenticated_feed_requires_host_allowlist(self):
        source = c.SourceConfig("feed", "private", "https://example.invalid", bearer_token_env="CUSTOM_FEED_TOKEN")
        with mock.patch.object(c.urllib.request, "build_opener") as opener:
            with self.assertRaisesRegex(ValueError, "FEED_ALLOWED_HOSTS"):
                c.fetch_feed_bytes(source)
        opener.assert_not_called()

    def test_feed_redirect_is_rejected_before_header_forwarding(self):
        handler = c.NoFeedRedirect()
        with self.assertRaisesRegex(ValueError, "redirects"):
            handler.redirect_request(None, None, 302, "redirect", {}, "https://example.invalid")

    def test_all_source_failure_does_not_write_data(self):
        with self.assertRaisesRegex(RuntimeError, "All configured sources failed"):
            self.run_collect(failure=RuntimeError("offline"))
        self.last_writer.assert_not_called()

    def test_explicit_overlap_uses_last_successful_watermark(self):
        os.environ["INCREMENTAL_OVERLAP_HOURS"] = "24"
        now = dt.datetime(2026, 11, 3, tzinfo=dt.timezone.utc)
        cutoff, mode = c.collection_cutoff({"generated_at_iso": now.isoformat(),
                                           "last_successful_collection_at": "2026-11-02T00:00:00+00:00"}, now, 7, True)
        self.assertEqual(cutoff, dt.datetime(2026, 11, 1, tzinfo=dt.timezone.utc))
        self.assertEqual(mode, "incremental")

    def test_deepseek_only_does_not_enable_other_keys(self):
        os.environ.update(LLM_PROVIDER="deepseek", OPENAI_API_KEY="dummy", LLM_API_KEY="dummy")
        self.assertFalse(c.llm_enabled())

    def test_priority_changes_ranking_not_relevance(self):
        paper = self.paper()
        low = c.Topic("low", "low", self.topic.description, self.topic.keywords, self.topic.arxiv_categories, .25)
        high_match, low_match = c.score_paper(self.topic, paper), c.score_paper(low, paper)
        self.assertEqual(high_match["score"], low_match["score"])
        self.assertGreater(high_match["ranking_score"], low_match["ranking_score"])

    def test_vlm_keyword_match_beats_higher_category_only_efficiency_score(self):
        vlm = c.Topic("vlm_core", "VLM", "", [
            "vision-language model", "visual reasoning", "multimodal reasoning", "image understanding",
            "video understanding", "instruction tuning", "spatial understanding", "visual question answering",
        ], ["cs.CV", "cs.CL", "cs.AI", "cs.HC"], 1.6)
        efficiency = c.Topic("vlm_efficiency", "Efficiency", "", [
            "visual token pruning", "visual token compression", "visual token selection", "visual token merging",
            "multimodal token compression", "efficient vision-language", "VLM acceleration", "adaptive visual resolution",
        ], ["cs.CV", "cs.CL", "cs.LG"], 1.8)
        self.config["topics"] = [efficiency.__dict__, vlm.__dict__]
        paper = self.paper("SMILE: General recognition")
        paper["summary"] = "The vision-language model aligns images and captions for general recognition. " * 5
        paper["categories"] = ["cs.CV", "cs.CL", "cs.LG"]
        vlm_match, efficiency_match = c.score_paper(vlm, paper), c.score_paper(efficiency, paper)
        self.assertGreater(efficiency_match["score"], vlm_match["score"])
        self.assertFalse(efficiency_match["keyword_hits"])
        self.assertEqual(vlm_match["keyword_hits"], ["vision-language model"])
        result, _ = self.run_collect([paper], max_summaries=0)
        best = result["papers"][0]["best_match"]
        self.assertEqual(best["topic_id"], "vlm_core")
        self.assertEqual(best["score"], vlm_match["score"])

    def test_daily_topic_caps_and_priority(self):
        nav = c.Topic("nav", "Navigation", "", [], [], .25, 1)
        papers = []
        for name, topic, score in [("nav1", nav, .9), ("gui", self.topic, .4), ("nav2", nav, .8)]:
            papers.append({"id": name, "best_match": {"topic_id": topic.id, "score": score, "ranking_score": score*topic.priority}})
        selected = c.select_daily_papers(papers, [self.topic, nav], 10)
        self.assertEqual([p["id"] for p in selected], ["gui", "nav1"])

    def test_unchanged_abstract_reuses_summary_without_paid_call(self):
        os.environ.update(LLM_PROVIDER="deepseek", DEEPSEEK_API_KEY="dummy", LLM_MODEL="deepseek-test")
        paper = self.paper()
        old = copy.deepcopy(paper)
        old.update(summary_status="llm", summary_fingerprint=c.summary_fingerprint(self.topic, paper),
                   best_match=c.score_paper(self.topic, paper), chinese_summary={"problem": "cached"})
        previous = {"papers": [old]}
        with mock.patch.object(c, "call_openai_compatible") as llm:
            result, _ = self.run_collect([paper], previous)
        llm.assert_not_called()
        self.assertEqual(result["papers"][0]["summary_status"], "cache")
        self.assertEqual(result["stats"]["summary_cache_hit_count"], 1)

    def test_changed_abstract_gets_new_summary_and_status(self):
        os.environ.update(LLM_PROVIDER="deepseek", DEEPSEEK_API_KEY="dummy", LLM_MODEL="deepseek-test")
        paper = self.paper()
        old = copy.deepcopy(paper)
        old.update(summary_status="llm", summary_fingerprint=c.summary_fingerprint(self.topic, paper),
                   best_match=c.score_paper(self.topic, paper), chinese_summary={"problem": "cached"})
        paper["summary"] += " New evidence."
        valid = {k: "Verified test summary" for k in ("problem", "method", "innovation", "evidence", "limitations", "why_relevant")}
        with mock.patch.object(c, "call_openai_compatible", return_value=valid) as llm:
            result, _ = self.run_collect([paper], {"papers": [old]})
        self.assertEqual(llm.call_count, 1)
        self.assertEqual(result["papers"][0]["summary_status"], "llm")
        self.assertEqual(result["stats"]["llm_success_count"], 1)

    def test_failed_summary_is_visible_and_does_not_claim_success(self):
        os.environ.update(LLM_PROVIDER="deepseek", DEEPSEEK_API_KEY="dummy")
        with mock.patch.object(c, "call_openai_compatible", side_effect=ValueError("synthetic failure")):
            result, _ = self.run_collect([self.paper()])
        self.assertEqual(result["papers"][0]["summary_status"], "failed")
        self.assertEqual(result["stats"]["llm_failure_count"], 1)
        self.assertEqual(result["stats"]["llm_success_count"], 0)

    def test_arxiv_revisions_dedupe_to_latest(self):
        old = self.paper()
        new = copy.deepcopy(old)
        new["id"] = "2611.00001v2"
        new["updated"] = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=1)).isoformat()
        self.assertEqual([p["id"] for p in c.dedupe_papers([old, new])], [new["id"]])

    def test_known_overlap_does_not_displace_new_paper_or_reset_first_seen(self):
        os.environ["INCREMENTAL_OVERLAP_HOURS"] = "24"
        old, new = self.paper(), self.paper()
        old["first_seen_at"] = (dt.datetime.now(dt.timezone.utc)-dt.timedelta(days=2)).isoformat()
        old["best_match"] = c.score_paper(self.topic, old)
        new["id"] = "2611.00002v1"
        new["categories"] = []  # The known paper deliberately has a higher rank.
        previous = {"generated_at_iso": old["published"], "papers": [copy.deepcopy(old)]}
        result, _ = self.run_collect([old, new], previous, max_summaries=0, max_new_papers=1)
        by_id = {p["id"]: p for p in result["papers"]}
        self.assertEqual(set(by_id), {old["id"], new["id"]})
        self.assertEqual(result["stats"]["new_paper_count"], 1)
        self.assertEqual(result["stats"]["refreshed_paper_count"], 1)
        self.assertEqual(by_id[old["id"]]["first_seen_at"], old["first_seen_at"])

    def test_new_paper_gets_paid_summary_before_higher_ranked_backlog(self):
        os.environ.update(LLM_PROVIDER="deepseek", DEEPSEEK_API_KEY="dummy", LLM_MODEL="deepseek-flash")
        old, new = self.paper(), self.paper("New GUI agent")
        old["best_match"] = c.score_paper(self.topic, old)
        new["id"], new["categories"] = "2611.00002v1", []
        valid = {k: "Test summary" for k in ("problem", "method", "innovation", "evidence", "limitations", "why_relevant")}
        with mock.patch.object(c, "call_openai_compatible", return_value=valid) as llm:
            result, _ = self.run_collect([old, new], {"papers": [copy.deepcopy(old)]}, max_summaries=1, max_new_papers=1)
        self.assertEqual(llm.call_count, 1)
        by_id = {p["id"]: p for p in result["papers"]}
        self.assertEqual(by_id[new["id"]]["summary_status"], "llm")
        self.assertEqual(by_id[old["id"]]["summary_status"], "fallback")
        self.assertEqual(result["stats"]["llm_attempt_count"], 1)

    def test_future_paper_is_excluded_before_selection(self):
        paper = self.paper()
        paper["updated"] = (dt.datetime.now(dt.timezone.utc)+dt.timedelta(days=1)).isoformat()
        result, _ = self.run_collect([paper])
        self.assertEqual(result["papers"], [])
        self.assertEqual(result["stats"]["future_filtered_count"], 1)
        self.assertEqual(result["stats"]["new_paper_count"], 0)

    def test_arxiv_query_limits_submitted_date(self):
        response = io.BytesIO(b'<feed xmlns="http://www.w3.org/2005/Atom"/>')
        with mock.patch.object(c.urllib.request, "urlopen", return_value=response) as request:
            self.assertEqual(c.fetch_arxiv_query('all:"GUI agent"', 50, "lastUpdatedDate", "descending", "GUI"), [])
        query = c.urllib.parse.parse_qs(c.urllib.parse.urlsplit(request.call_args.args[0].full_url).query)
        self.assertRegex(query["search_query"][0], r'submittedDate:\[199101010000 TO \d{12}\]')

    def test_refresh_of_old_low_match_preserves_original_first_seen(self):
        original = self.paper()
        original["first_seen_at"] = "2025-01-01T00:00:00+00:00"
        original["best_match"] = {"level": "low"}
        updated = copy.deepcopy(original)
        updated.pop("first_seen_at")
        result, _ = c.merge_with_retained_papers([updated], {"papers": [original]}, dt.datetime.now(dt.timezone.utc), 45)
        self.assertEqual(result[0]["first_seen_at"], original["first_seen_at"])


if __name__ == "__main__":
    unittest.main()
