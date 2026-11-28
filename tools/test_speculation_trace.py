"""Stdlib tests for the MTP speculation trace schema, reducer and fixtures."""
import copy
import json
import unittest

from speculation_trace import (MAX_EVENTS, SCHEMA, TraceCollector, TraceError,
                               synthetic_fixtures, summarize, validate_trace)


def fixture(name):
    return copy.deepcopy(synthetic_fixtures()[name])


def resequence(events):
    for index, event in enumerate(events):
        event["seq"] = index
    return events


class FixtureTests(unittest.TestCase):
    def test_required_cases_exist(self):
        self.assertEqual(set(synthetic_fixtures()), {
            "d1_all_accepted_bonus", "d1_first_rejection_unexamined",
            "d2_later_rejection", "d1_eos", "d2_short_budget",
            "d1_zero_output", "d0_target_only"})

    def test_all_fixtures_validate_and_are_json_safe(self):
        summaries = {}
        for name, events in synthetic_fixtures().items():
            summaries[name] = summarize(events, synthetic=True)
            self.assertEqual(summaries[name]["schema"], SCHEMA)
            self.assertTrue(summaries[name]["timing"]["synthetic"])
            self.assertEqual(summaries[name]["timing"]["source"], "fake_clock")
        encoded = json.dumps(summaries, sort_keys=True, allow_nan=False)
        self.assertGreater(len(encoded), 1000)

    def test_all_accepted_plus_bonus(self):
        s = summarize(fixture("d1_all_accepted_bonus"))
        self.assertEqual((s["rounds"], s["proposed_tokens"], s["verified_tokens"],
                          s["accepted_tokens"], s["rejected_tokens"],
                          s["unexamined_tokens"], s["correction_tokens"],
                          s["bonus_tokens"], s["committed_tokens"]),
                         (1, 2, 2, 2, 0, 0, 0, 1, 3))
        self.assertEqual(s["acceptance_per_verified"], 1.0)
        self.assertEqual(s["accepted_per_proposed"], 1.0)
        self.assertEqual(s["work"]["target_calls"], 3)
        self.assertEqual(s["work"]["target_tokens"], 7)
        self.assertEqual(s["work"]["shallow_calls"], 3)
        self.assertEqual(s["work"]["shallow_tokens"], 8)
        self.assertEqual(s["work"]["target_decode_tokens"], 3)
        self.assertEqual(s["work"]["target_tokens_per_committed_token"], 1.0)
        self.assertEqual(s["work"]["target_extra_tokens_over_committed"], 0)
        self.assertAlmostEqual(s["timing"]["ttft_seconds"], 0.015)
        self.assertAlmostEqual(s["timing"]["decode_seconds"], 0.001)
        self.assertAlmostEqual(s["timing"]["total_seconds"], 0.016)
        self.assertAlmostEqual(s["timing"]["prefill_seconds"], 0.004)
        self.assertIn("target:target_bonus_advance", s["timing"]["phase_seconds"])

    def test_first_rejection_unexamined_suffix(self):
        s = summarize(fixture("d1_first_rejection_unexamined"))
        self.assertEqual((s["rounds"], s["proposed_tokens"], s["verified_tokens"],
                          s["accepted_tokens"], s["rejected_tokens"],
                          s["unexamined_tokens"], s["correction_tokens"],
                          s["bonus_tokens"], s["committed_tokens"]),
                         (2, 3, 2, 1, 1, 1, 1, 0, 2))
        self.assertEqual(s["acceptance_per_verified"], 0.5)
        self.assertAlmostEqual(s["accepted_per_proposed"], 1 / 3)
        self.assertAlmostEqual(s["unexamined_per_proposed"], 1 / 3)
        self.assertEqual([row["offset"] for row in s["per_offset"]], [1, 2])
        self.assertEqual(s["per_offset"][0]["accepted"], 1)
        self.assertEqual(s["per_offset"][0]["rejected"], 1)
        self.assertEqual(s["per_offset"][1]["unexamined"], 1)
        self.assertEqual(s["per_offset"][1]["verified"], 0)
        self.assertIsNone(s["per_offset"][1]["acceptance_per_verified"])
        self.assertEqual(s["work"]["target_calls"], 4)
        self.assertEqual(s["work"]["target_tokens"], 8)
        self.assertEqual(s["work"]["shallow_calls"], 5)

    def test_later_rejection_and_replay_amplification(self):
        s = summarize(fixture("d2_later_rejection"))
        self.assertEqual((s["proposed_tokens"], s["accepted_tokens"],
                          s["rejected_tokens"], s["unexamined_tokens"],
                          s["correction_tokens"], s["committed_tokens"]),
                         (2, 1, 1, 0, 1, 2))
        self.assertEqual(s["acceptance_per_verified"], 0.5)
        self.assertEqual(s["work"]["target_calls"], 3)
        self.assertEqual(s["work"]["target_prefill_tokens"], 4)
        self.assertEqual(s["work"]["target_decode_tokens"], 4)
        self.assertEqual(s["work"]["target_tokens_per_committed_token"], 2.0)
        self.assertEqual(s["work"]["shallow_calls"], 3)
        self.assertEqual(s["work"]["shallow_tokens"], 6)

    def test_eos_and_short_budget(self):
        eos = summarize(fixture("d1_eos"))
        self.assertTrue(eos["eos_committed"])
        self.assertEqual((eos["proposed_tokens"], eos["accepted_tokens"],
                          eos["bonus_tokens"], eos["committed_tokens"]),
                         (1, 1, 0, 1))
        short = summarize(fixture("d2_short_budget"))
        self.assertFalse(short["eos_committed"])
        self.assertEqual((short["proposed_tokens"], short["accepted_tokens"],
                          short["bonus_tokens"], short["committed_tokens"]),
                         (1, 1, 0, 1))

    def test_zero_output_has_null_ttft(self):
        s = summarize(fixture("d1_zero_output"))
        self.assertEqual((s["rounds"], s["committed_tokens"], s["proposed_tokens"]),
                         (0, 0, 0))
        self.assertIsNone(s["timing"]["ttft_seconds"])
        self.assertIsNone(s["timing"]["decode_seconds"])
        self.assertIsNone(s["acceptance_per_verified"])
        self.assertIsNone(s["accepted_per_proposed"])
        self.assertIsNone(s["work"]["target_tokens_per_committed_token"])
        self.assertEqual(s["work"]["target_calls"], 1)
        self.assertEqual(s["work"]["shallow_calls"], 1)

    def test_d0_has_no_shallow_work_or_proposals(self):
        s = summarize(fixture("d0_target_only"))
        self.assertEqual((s["strategy"], s["rounds"], s["committed_tokens"]),
                         ("D0", 2, 2))
        self.assertEqual((s["proposed_tokens"], s["accepted_tokens"],
                          s["rejected_tokens"], s["correction_tokens"],
                          s["bonus_tokens"]), (0, 0, 0, 0, 0))
        self.assertEqual(s["work"]["shallow_calls"], 0)
        self.assertEqual(s["work"]["target_calls"], 3)
        self.assertEqual(s["work"]["target_tokens"], 6)
        self.assertEqual(s["work"]["target_tokens_per_committed_token"], 1.0)
        self.assertEqual(s["per_offset"], [])

    def test_collector_bounded(self):
        collector = TraceCollector()
        collector.events = [None] * MAX_EVENTS
        with self.assertRaises(TraceError):
            collector(object())


class MalformedTests(unittest.TestCase):
    def assert_rejected(self, events):
        with self.assertRaises(TraceError):
            summarize(events)

    def test_top_level_shape(self):
        self.assert_rejected(None)
        self.assert_rejected([])
        self.assert_rejected([1])
        self.assert_rejected([None] * (MAX_EVENTS + 1))

    def test_wrong_schema_and_seq(self):
        events = fixture("d1_all_accepted_bonus")
        events[0]["schema"] = "other"
        self.assert_rejected(events)
        events = fixture("d1_all_accepted_bonus")
        events[3]["seq"] += 1
        self.assert_rejected(events)

    def test_timestamp_decrease(self):
        events = fixture("d1_all_accepted_bonus")
        events[5]["timestamp_ns"] = events[4]["timestamp_ns"] - 1
        self.assert_rejected(events)

    def test_unknown_kind_and_content_field(self):
        events = fixture("d1_all_accepted_bonus")
        events[2]["kind"] = "token"
        self.assert_rejected(events)
        events = fixture("d1_all_accepted_bonus")
        events[2]["token_ids"] = [1, 2, 3]
        self.assert_rejected(events)
        events = fixture("d1_all_accepted_bonus")
        events[2]["prompt"] = "secret text"
        self.assert_rejected(events)

    def test_missing_or_reordered_prefill(self):
        events = fixture("d1_all_accepted_bonus")
        del events[1:3]
        resequence(events)
        self.assert_rejected(events)
        events = fixture("d1_all_accepted_bonus")
        events[3], events[5] = events[5], events[3]
        events[4], events[6] = events[6], events[4]
        resequence(events)
        self.assert_rejected(events)

    def test_phase_count_mismatch(self):
        events = fixture("d1_all_accepted_bonus")
        events[7]["actual_tokens"] = 1
        events[7]["after_length"] = events[7]["before_length"] + 1
        self.assert_rejected(events)

    def test_draft_after_rejection_must_be_unexamined(self):
        events = fixture("d1_first_rejection_unexamined")
        events[10]["status"] = "accepted"
        events[10]["verified"] = True
        self.assert_rejected(events)

    def test_duplicate_rejection(self):
        events = fixture("d2_later_rejection")
        events[9]["status"] = "rejected"
        self.assert_rejected(events)

    def test_commit_arithmetic_and_lengths(self):
        def commit():
            events = fixture("d2_later_rejection")
            return events, next(e for e in events if e["kind"] == "round_commit")
        events, event = commit()
        event["committed"] = 3
        self.assert_rejected(events)
        events, event = commit()
        event["target_after"] += 1
        self.assert_rejected(events)
        events, event = commit()
        event["output_after"] += 1
        self.assert_rejected(events)

    def test_eos_without_flag_and_early_end(self):
        events = fixture("d1_eos")
        events[0]["eos"] = False
        self.assert_rejected(events)
        events = fixture("d1_all_accepted_bonus")
        events[0]["max_new_tokens"] = 4
        self.assert_rejected(events)

    def test_unexamined_requires_rejection_or_eos(self):
        events = fixture("d1_all_accepted_bonus")
        events[0]["eos"] = True
        events[0]["max_new_tokens"] = 4
        events[9]["status"] = "unexamined"
        events[9]["verified"] = False
        # Counts still reconcile only if the commit is adjusted; expected error
        # is the unexamined-without-rejection/EOS rule or an earlier mismatch.
        self.assert_rejected(events)

    def test_d0_proposal_is_invalid(self):
        events = fixture("d0_target_only")
        events[6]["proposed"] = 1
        self.assert_rejected(events)

    def test_run_end_totals(self):
        events = fixture("d2_later_rejection")
        events[-1]["proposed"] = 3
        self.assert_rejected(events)
        events = fixture("d1_zero_output")
        events[-1]["output_length"] = 1
        self.assert_rejected(events)


class ApiTests(unittest.TestCase):
    def test_validate_trace_returns_summary(self):
        events = fixture("d1_eos")
        self.assertEqual(validate_trace(events), summarize(events))

    def test_event_values_are_scalar(self):
        for events in synthetic_fixtures().values():
            for event in events:
                for key, value in event.items():
                    self.assertIn(type(value), (int, float, bool, str), (key, value))


if __name__ == "__main__":
    unittest.main(verbosity=2)
