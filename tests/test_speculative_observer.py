"""Hosted-CPU integration tests for the opt-in MTP speculation observer."""
import unittest

import torch
from splot import MTPLM, RecurrentConfig, ModelRunner, Prefix, generate
from splot.speculative import TRACE_SCHEMA
from speculation_trace import TraceCollector, run_d0, summarize


class Toy:
    """Two-token vocabulary; target next-token probabilities depend on last token."""
    def __init__(self, target=False, forced=None):
        self.target = target
        self.forced = forced
        self.calls = []

    def logits(self, tokens):
        if self.forced is not None:
            return torch.tensor(self.forced, dtype=torch.float64).log()
        p = ([.8, .2] if tokens[-1] == 0 else [.35, .65]) if self.target else ([.25, .75] if tokens[-1] == 0 else [.9, .1])
        return torch.tensor(p, dtype=torch.float64).log()

    def advance(self, prefix, tokens):
        old = () if prefix is None else prefix.cache
        self.calls.append((old, tuple(tokens)))
        seq = old
        out = []
        for t in tokens:
            seq = seq + (t,)
            out.append(self.logits(seq))
        return Prefix(len(seq), seq, out[-1], torch.tensor([.6, .4], dtype=torch.float64).log()), torch.stack(out)


class Clock:
    def __init__(self, step=1_000_000):
        self.value = 1_000_000_000
        self.step = step

    def __call__(self):
        value = self.value
        self.value += self.step
        return value


def toy_prefix_equal(test, expected, actual):
    test.assertEqual(expected.length, actual.length)
    test.assertEqual(expected.cache, actual.cache)
    torch.testing.assert_close(expected.next_logits, actual.next_logits, atol=0, rtol=0)
    if expected.offset2_logits is not None or actual.offset2_logits is not None:
        torch.testing.assert_close(expected.offset2_logits, actual.offset2_logits, atol=0, rtol=0)


def model_prefix_equal(test, expected, actual):
    test.assertEqual(expected.length, actual.length)
    for (ka, sa), (kb, sb) in zip(expected.cache.states, actual.cache.states):
        test.assertEqual(ka, kb)
        for a, b in zip(sa[0], sb[0]):
            torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(expected.next_logits, actual.next_logits, atol=1e-12, rtol=1e-12)
    if expected.offset2_logits is not None or actual.offset2_logits is not None:
        torch.testing.assert_close(expected.offset2_logits, actual.offset2_logits, atol=1e-12, rtol=1e-12)


class ObserverIdentityTests(unittest.TestCase):
    def test_toy_off_on_identity_and_rng(self):
        for strategy in ("D1", "D2"):
            for greedy in (False, True):
                target_a, shallow_a = Toy(True), Toy()
                rng_a = torch.Generator().manual_seed(1234)
                baseline = generate(target_a, shallow_a, [0], 5, strategy=strategy,
                                    draft_length=2, greedy=greedy, rng=rng_a)
                target_b, shallow_b = Toy(True), Toy()
                rng_b = torch.Generator().manual_seed(1234)
                collector = TraceCollector()
                observed = generate(target_b, shallow_b, [0], 5, strategy=strategy,
                                    draft_length=2, greedy=greedy, rng=rng_b,
                                    observer=collector, clock=Clock())
                self.assertEqual(baseline.tokens, observed.tokens)
                self.assertEqual((baseline.proposed, baseline.accepted, baseline.rejected, baseline.rounds),
                                 (observed.proposed, observed.accepted, observed.rejected, observed.rounds))
                toy_prefix_equal(self, baseline.target, observed.target)
                toy_prefix_equal(self, baseline.shallow, observed.shallow)
                self.assertTrue(torch.equal(rng_a.get_state(), rng_b.get_state()))
                self.assertEqual(target_a.calls, target_b.calls)
                self.assertEqual(shallow_a.calls, shallow_b.calls)
                summary = summarize(collector.events)
                self.assertEqual(summary["strategy"], strategy)
                self.assertEqual(summary["committed_tokens"], len(baseline.tokens))
                self.assertIsNotNone(summary["timing"]["ttft_seconds"])

    def test_fresh_prefix_matches_returned_cache_and_logits(self):
        target, shallow = Toy(True), Toy()
        collector = TraceCollector()
        result = generate(target, shallow, [0], 4, strategy="D2", greedy=False,
                          rng=torch.Generator().manual_seed(7),
                          observer=collector, clock=Clock())
        summarize(collector.events)
        for runner, prefix in ((target, result.target), (shallow, result.shallow)):
            fresh, _ = runner.advance(None, [0] + list(result.tokens))
            toy_prefix_equal(self, fresh, prefix)

    def test_event_content_is_scalar_and_schema_stable(self):
        collector = TraceCollector()
        generate(Toy(True), Toy(), [0], 3, strategy="D1", greedy=True,
                 observer=collector, clock=Clock())
        for event in collector.events:
            self.assertEqual(event["schema"], TRACE_SCHEMA)
            for value in event.values():
                self.assertIn(type(value), (int, float, bool, str))


class ScenarioTests(unittest.TestCase):
    def test_all_accepted_bonus_and_calls(self):
        target, shallow = Toy(forced=[1., 0.]), Toy(forced=[1., 0.])
        collector = TraceCollector()
        result = generate(target, shallow, [1], 3, strategy="D1", greedy=True,
                          observer=collector, clock=Clock())
        summary = summarize(collector.events)
        self.assertEqual(result.tokens, (0, 0, 0))
        self.assertEqual((summary["proposed_tokens"], summary["accepted_tokens"],
                          summary["rejected_tokens"], summary["unexamined_tokens"],
                          summary["bonus_tokens"], summary["committed_tokens"]),
                         (2, 2, 0, 0, 1, 3))
        self.assertEqual(summary["work"]["target_calls"], 3)
        self.assertEqual(target.calls, [((), (1,)), ((1,), (0, 0)), ((1, 0, 0), (0,))])

    def test_first_rejection_unexamined_suffix(self):
        class Target(Toy):
            def logits(self, tokens):
                return torch.tensor([0., 1.] if len(tokens) == 1 else [1., 0.],
                                    dtype=torch.float64).log()
        target, shallow = Target(), Toy(forced=[1., 0.])
        collector = TraceCollector()
        result = generate(target, shallow, [0], 2, strategy="D2", greedy=True,
                          observer=collector, clock=Clock())
        summary = summarize(collector.events)
        self.assertEqual(result.tokens, (1, 0))
        self.assertEqual((summary["proposed_tokens"], summary["accepted_tokens"],
                          summary["rejected_tokens"], summary["unexamined_tokens"],
                          summary["correction_tokens"], summary["bonus_tokens"],
                          summary["committed_tokens"]), (3, 1, 1, 1, 1, 0, 2))
        self.assertEqual(summary["per_offset"][0]["rejected"], 1)
        self.assertEqual(summary["per_offset"][1]["unexamined"], 1)
        self.assertEqual(summary["work"]["target_calls"], 4)  # prefill + verify + replay + second verify

    def test_later_rejection(self):
        class Target(Toy):
            def logits(self, tokens):
                return torch.tensor([1., 0.] if len(tokens) == 1 else [0., 1.],
                                    dtype=torch.float64).log()
        target, shallow = Target(), Toy(forced=[1., 0.])
        collector = TraceCollector()
        result = generate(target, shallow, [0], 2, strategy="D2", greedy=True,
                          observer=collector, clock=Clock())
        summary = summarize(collector.events)
        self.assertEqual(result.tokens, (0, 1))
        self.assertEqual((summary["accepted_tokens"], summary["rejected_tokens"],
                          summary["unexamined_tokens"], summary["correction_tokens"],
                          summary["committed_tokens"]), (1, 1, 0, 1, 2))
        self.assertEqual(summary["work"]["target_tokens_per_committed_token"], 2.0)

    def test_eos_and_zero_budget(self):
        target, shallow = Toy(forced=[0., 1.]), Toy(forced=[0., 1.])
        collector = TraceCollector()
        result = generate(target, shallow, [0], 5, strategy="D1", eos=1,
                          greedy=True, observer=collector, clock=Clock())
        summary = summarize(collector.events)
        self.assertEqual(result.tokens, (1,))
        self.assertTrue(summary["eos_committed"])
        self.assertEqual((summary["proposed_tokens"], summary["accepted_tokens"],
                          summary["unexamined_tokens"], summary["committed_tokens"]),
                         (1, 1, 0, 1))

        collector = TraceCollector()
        result = generate(Toy(True), Toy(), [0], 0, strategy="D2", greedy=True,
                          observer=collector, clock=Clock())
        summary = summarize(collector.events)
        self.assertEqual((result.tokens, summary["rounds"], summary["committed_tokens"]), ((), 0, 0))
        self.assertIsNone(summary["timing"]["ttft_seconds"])
        self.assertIsNone(summary["timing"]["decode_seconds"])

    def test_short_final_draft(self):
        collector = TraceCollector()
        result = generate(Toy(forced=[1., 0.]), Toy(forced=[1., 0.]), [1], 1,
                          strategy="D2", greedy=True, observer=collector, clock=Clock())
        summary = summarize(collector.events)
        self.assertEqual((result.tokens, summary["proposed_tokens"],
                          summary["accepted_tokens"], summary["bonus_tokens"]),
                         ((0,), 1, 1, 0))


class D0Tests(unittest.TestCase):
    def test_d0_greedy_matches_d1_d2_on_actual_model(self):
        torch.manual_seed(42)
        model = MTPLM(RecurrentConfig()).double().eval()
        target = ModelRunner(model, 4)
        shallow = ModelRunner(model, 1, offset2=True)
        prompt = [1, 80, 81, 82]
        budget = 5
        expected = []
        state, _ = target.advance(None, prompt)
        for _ in range(budget):
            token = int(state.next_logits.argmax())
            expected.append(token)
            state, _ = target.advance(state, [token])

        d1 = generate(target, shallow, prompt, budget, strategy="D1", greedy=True)
        d2 = generate(target, shallow, prompt, budget, strategy="D2", greedy=True)
        d0_collector = TraceCollector()
        d0 = run_d0(target, prompt, budget, greedy=True,
                    observer=d0_collector, clock=Clock())
        self.assertEqual(list(d1.tokens), expected)
        self.assertEqual(list(d2.tokens), expected)
        self.assertEqual(list(d0.tokens), expected)
        self.assertEqual(d0.target.length, len(prompt) + budget)
        self.assertEqual(d0.rounds, budget)
        fresh, _ = target.advance(None, prompt + list(d0.tokens))
        model_prefix_equal(self, fresh, d0.target)
        summary = summarize(d0_collector.events)
        self.assertEqual((summary["strategy"], summary["proposed_tokens"],
                          summary["committed_tokens"]), ("D0", 0, budget))
        self.assertEqual(summary["work"]["shallow_calls"], 0)
        self.assertEqual(summary["work"]["target_calls"], budget + 1)

    def test_d0_eos_and_zero_budget(self):
        target = Toy(forced=[0., 1.])
        collector = TraceCollector()
        result = run_d0(target, [0], 5, eos=1, greedy=True,
                        observer=collector, clock=Clock())
        summary = summarize(collector.events)
        self.assertEqual((result.tokens, result.rounds), ((1,), 1))
        self.assertTrue(summary["eos_committed"])
        collector = TraceCollector()
        result = run_d0(Toy(True), [0], 0, greedy=True,
                        observer=collector, clock=Clock())
        summary = summarize(collector.events)
        self.assertEqual((result.tokens, result.rounds), ((), 0))
        self.assertIsNone(summary["timing"]["ttft_seconds"])

    def test_d0_stochastic_returns_valid_tokens(self):
        target = Toy(forced=[.5, .5])
        result = run_d0(target, [0], 3, rng=torch.Generator().manual_seed(5))
        self.assertEqual(len(result.tokens), 3)
        self.assertTrue(set(result.tokens) <= {0, 1})


class ActualModelObserverIdentityTests(unittest.TestCase):
    def test_observer_off_on_model_cache_and_rng(self):
        torch.manual_seed(42)
        model = MTPLM(RecurrentConfig()).double().eval()
        target = ModelRunner(model, 4)
        shallow = ModelRunner(model, 1, offset2=True)
        prompt = [1, 80, 81, 82]
        rng_a = torch.Generator().manual_seed(9)
        baseline = generate(target, shallow, prompt, 4, strategy="D1", greedy=False, rng=rng_a)
        rng_b = torch.Generator().manual_seed(9)
        collector = TraceCollector()
        observed = generate(target, shallow, prompt, 4, strategy="D1", greedy=False,
                            rng=rng_b, observer=collector, clock=Clock())
        self.assertEqual(baseline.tokens, observed.tokens)
        self.assertTrue(torch.equal(rng_a.get_state(), rng_b.get_state()))
        model_prefix_equal(self, baseline.target, observed.target)
        model_prefix_equal(self, baseline.shallow, observed.shallow)
        summary = summarize(collector.events)
        self.assertEqual(summary["committed_tokens"], len(observed.tokens))
        self.assertGreaterEqual(summary["work"]["target_calls"], 2)
        self.assertGreaterEqual(summary["work"]["shallow_calls"], 2)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
