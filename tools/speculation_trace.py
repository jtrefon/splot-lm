"""Exact MTP speculation trace schema, reducer and target-only D0 baseline.

The reducer, schema validation and synthetic fixtures are stdlib-only. The
optional :func:`run_d0` helper imports ``torch`` and ``splot.speculative`` only
when called, so this module can be tested on a machine without torch.

Trace events contain counts, lengths, role/phase names, booleans and monotonic
nanosecond timestamps. They must never contain token IDs, text, logits, caches
or tensors. ``summarize`` is the single source for acceptance, coverage,
work-count, replay-amplification and timing metrics used by the observer tests
and the future trained-checkpoint benchmark.
"""
from dataclasses import dataclass
import math
import time

SCHEMA = "splot.mtp.trace.v2"
MAX_EVENTS = 100000
STRATEGIES = ("D0", "D1", "D2")
STATUSES = ("accepted", "rejected", "unexamined")
TARGET_PHASES = ("target_prefill", "target_verify", "target_rejection_replay",
                 "target_bonus_advance", "target_step")
SHALLOW_PHASES = ("shallow_prefill", "shallow_draft", "shallow_committed_replay")
PREFILL_PHASES = ("target_prefill", "shallow_prefill")

_COMMON = {"schema", "seq", "kind", "timestamp_ns"}
_FIELDS = {
    "run_start": {"strategy", "draft_length", "max_new_tokens", "prompt_length",
                  "greedy", "temperature", "eos"},
    "phase_start": {"role", "phase", "round", "before_length"},
    "phase_end": {"role", "phase", "round", "before_length", "after_length",
                  "requested_tokens", "actual_tokens", "model_calls"},
    "draft": {"round", "offset", "status", "proposed", "verified"},
    "round_commit": {"round", "strategy", "proposed", "verified", "accepted",
                     "rejected", "unexamined", "correction", "bonus",
                     "committed", "target_before", "target_after",
                     "shallow_before", "shallow_after", "output_before",
                     "output_after", "eos_committed"},
    "run_end": {"rounds", "proposed", "accepted", "rejected", "output_length",
                "target_length", "shallow_length"},
}
_KINDS = set(_FIELDS)


class TraceError(ValueError):
    """Malformed, out-of-order, oversized or content-bearing trace."""


class TraceCollector:
    """Callable observer that stores a bounded trace for later validation."""

    def __init__(self):
        self.events = []

    def __call__(self, event):
        if len(self.events) >= MAX_EVENTS:
            raise TraceError(f"trace exceeds {MAX_EVENTS} events")
        self.events.append(event)

    def summarize(self):
        return summarize(self.events)


def _int(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise TraceError(f"{name} must be an integer >= {minimum}")
    return value


def _bool(value, name):
    if type(value) is not bool:
        raise TraceError(f"{name} must be a boolean")
    return value


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise TraceError(f"{name} must be a finite number")
    return float(value)


def _phase_role(phase):
    if phase in TARGET_PHASES:
        return "target"
    if phase in SHALLOW_PHASES:
        return "shallow"
    raise TraceError(f"unknown phase: {phase}")


def _check_event(event, index):
    if not isinstance(event, dict):
        raise TraceError(f"event {index} must be an object")
    kind = event.get("kind")
    if kind not in _KINDS:
        raise TraceError(f"event {index} has unknown kind: {kind!r}")
    expected = _COMMON | _FIELDS[kind]
    if set(event) != expected:
        missing = expected - set(event)
        extra = set(event) - expected
        raise TraceError(f"event {index} fields mismatch; missing={sorted(missing)} extra={sorted(extra)}")
    if event["schema"] != SCHEMA:
        raise TraceError(f"event {index} schema mismatch")
    if _int(event["seq"], "seq") != index:
        raise TraceError(f"event {index} seq mismatch")
    _int(event["timestamp_ns"], "timestamp_ns")
    if kind == "run_start":
        if event["strategy"] not in STRATEGIES:
            raise TraceError("invalid strategy")
        draft = _int(event["draft_length"], "draft_length")
        _int(event["max_new_tokens"], "max_new_tokens")
        _int(event["prompt_length"], "prompt_length", 1)
        _bool(event["greedy"], "greedy")
        _bool(event["eos"], "eos")
        temperature = _number(event["temperature"], "temperature")
        if temperature <= 0:
            raise TraceError("temperature must be positive")
        if event["strategy"] == "D0" and draft != 0:
            raise TraceError("D0 draft_length must be zero")
        if event["strategy"] == "D1" and draft < 1:
            raise TraceError("D1 draft_length must be positive")
        if event["strategy"] == "D2" and draft != 2:
            raise TraceError("D2 draft_length must be two")
    elif kind in ("phase_start", "phase_end"):
        role = event["role"]
        phase = event["phase"]
        if role not in ("target", "shallow") or _phase_role(phase) != role:
            raise TraceError(f"invalid role/phase pair: {role}/{phase}")
        _int(event["round"], "round")
        before = _int(event["before_length"], "before_length")
        if kind == "phase_end":
            after = _int(event["after_length"], "after_length")
            requested = _int(event["requested_tokens"], "requested_tokens")
            actual = _int(event["actual_tokens"], "actual_tokens")
            calls = _int(event["model_calls"], "model_calls")
            if after < before:
                raise TraceError("phase after_length is below before_length")
            if actual != after - before:
                raise TraceError("phase actual_tokens does not equal after-before")
            if actual > requested:
                raise TraceError("phase actual_tokens exceeds requested_tokens")
            # A serial D1 scratch advance moves exactly one token per call; D2
            # performs zero draft calls. Every other phase is one model call.
            if phase == "shallow_draft":
                if calls != actual:
                    raise TraceError("shallow_draft model_calls must equal actual scratch advances")
            elif calls != 1:
                raise TraceError(f"{phase} must make exactly one model call per phase occurrence")
    elif kind == "draft":
        _int(event["round"], "round", 1)
        _int(event["offset"], "offset", 1)
        status = event["status"]
        if status not in STATUSES:
            raise TraceError("invalid draft status")
        if event["proposed"] is not True:
            raise TraceError("draft proposed must be true")
        _bool(event["verified"], "verified")
        if event["verified"] != (status != "unexamined"):
            raise TraceError("draft verified flag disagrees with status")
    elif kind == "round_commit":
        if event["strategy"] not in STRATEGIES:
            raise TraceError("invalid round strategy")
        _int(event["round"], "round", 1)
        for name in ("proposed", "verified", "accepted", "rejected",
                     "unexamined", "correction", "bonus", "committed",
                     "target_before", "target_after", "shallow_before",
                     "shallow_after", "output_before", "output_after"):
            _int(event[name], name)
        _bool(event["eos_committed"], "eos_committed")
    else:
        for name in ("rounds", "proposed", "accepted", "rejected",
                     "output_length", "target_length", "shallow_length"):
            _int(event[name], name)


def _phase_pair(events, cursor, role, phase, round_index, before=None):
    if cursor + 1 >= len(events):
        raise TraceError(f"missing phase pair {role}/{phase} round {round_index}")
    start, end = events[cursor], events[cursor + 1]
    if start["kind"] != "phase_start" or end["kind"] != "phase_end":
        raise TraceError(f"expected phase pair at event {cursor}: {role}/{phase}")
    for event in (start, end):
        if event["role"] != role or event["phase"] != phase or event["round"] != round_index:
            raise TraceError(f"phase order mismatch at event {cursor}: {role}/{phase} round {round_index}")
    if end["before_length"] != start["before_length"]:
        raise TraceError("phase pair before_length mismatch")
    if before is not None and start["before_length"] != before:
        raise TraceError(f"phase {phase} round {round_index} starts at {start['before_length']}, expected {before}")
    return start, end, cursor + 2


def _expect_phase(events, cursor, phase):
    if cursor >= len(events) or events[cursor]["kind"] != "phase_start" or events[cursor]["phase"] != phase:
        return False
    return True


def summarize(events, *, synthetic=False):
    """Validate a bounded trace and return aggregate counts, ratios and timings.

    Raises :class:`TraceError` for malformed schema, ordering, counts or any
    content-bearing field. Ratios are ``None`` when the denominator is zero.
    ``synthetic`` marks injected fake-clock fixtures; it never marks a measured
    run and does not change counts.
    """
    if not isinstance(events, list):
        raise TraceError("trace must be a list")
    if not 1 <= len(events) <= MAX_EVENTS:
        raise TraceError(f"trace length must be 1..{MAX_EVENTS}")
    for index, event in enumerate(events):
        _check_event(event, index)
        if index and event["timestamp_ns"] < events[index - 1]["timestamp_ns"]:
            raise TraceError(f"timestamp decreases at event {index}")
    if events[0]["kind"] != "run_start":
        raise TraceError("first event must be run_start")
    if events[-1]["kind"] != "run_end":
        raise TraceError("last event must be run_end")
    if sum(e["kind"] == "run_start" for e in events) != 1:
        raise TraceError("exactly one run_start is required")
    if sum(e["kind"] == "run_end" for e in events) != 1:
        raise TraceError("exactly one run_end is required")

    run = events[0]
    end = events[-1]
    strategy = run["strategy"]
    prompt = run["prompt_length"]
    budget = run["max_new_tokens"]
    eos_enabled = run["eos"]
    cursor = 1
    phase_pairs = []

    target_prefill_start, target_prefill, cursor = _phase_pair(events, cursor, "target", "target_prefill", 0, 0)
    if target_prefill["after_length"] != prompt or target_prefill["actual_tokens"] != prompt:
        raise TraceError("target prefill does not cover the prompt")
    phase_pairs.append(("target", "target_prefill", target_prefill_start["timestamp_ns"], target_prefill["timestamp_ns"], target_prefill))
    if strategy != "D0":
        shallow_prefill_start, shallow_prefill, cursor = _phase_pair(events, cursor, "shallow", "shallow_prefill", 0, 0)
        if shallow_prefill["after_length"] != prompt or shallow_prefill["actual_tokens"] != prompt:
            raise TraceError("shallow prefill does not cover the prompt")
        phase_pairs.append(("shallow", "shallow_prefill", shallow_prefill_start["timestamp_ns"], shallow_prefill["timestamp_ns"], shallow_prefill))

    target_len = shallow_len = prompt
    output_len = rounds = 0
    total_proposed = total_accepted = total_rejected = 0
    total_correction = total_bonus = total_unexamined = 0
    per_offset = {}
    first_commit = last_commit = None
    first_round_committed = 0
    eos_seen = False

    while cursor < len(events) - 1:
        if eos_seen:
            raise TraceError("events follow a committed EOS")
        if output_len >= budget:
            raise TraceError("events follow an exhausted token budget")
        round_index = rounds + 1
        if strategy == "D0":
            start, step, cursor = _phase_pair(events, cursor, "target", "target_step", round_index, target_len)
            if step["actual_tokens"] != 1:
                raise TraceError("D0 target_step must advance exactly one token")
            phase_pairs.append(("target", "target_step", start["timestamp_ns"], step["timestamp_ns"], step))
            if cursor >= len(events) or events[cursor]["kind"] != "round_commit":
                raise TraceError("missing D0 round_commit")
            commit = events[cursor]
            cursor += 1
            if commit["strategy"] != "D0" or commit["round"] != round_index:
                raise TraceError("D0 round_commit strategy/round mismatch")
            for name in ("proposed", "verified", "accepted", "rejected",
                         "unexamined", "correction", "bonus"):
                if commit[name] != 0:
                    raise TraceError(f"D0 round_commit {name} must be zero")
            if commit["committed"] != 1:
                raise TraceError("D0 round_commit must commit one token")
        else:
            draft_start, draft_end, cursor = _phase_pair(events, cursor, "shallow", "shallow_draft",
                                                         round_index, shallow_len)
            phase_pairs.append(("shallow", "shallow_draft", draft_start["timestamp_ns"], draft_end["timestamp_ns"], draft_end))
            if strategy == "D2" and (draft_end["actual_tokens"] != 0 or draft_end["requested_tokens"] != 0):
                raise TraceError("D2 shallow_draft must not request or advance a scratch cache")
            if draft_end["requested_tokens"] > run["draft_length"] - 1:
                raise TraceError("shallow_draft requested_tokens exceeds draft length minus one")

            verify_start, verify_end, cursor = _phase_pair(events, cursor, "target", "target_verify",
                                                           round_index, target_len)
            phase_pairs.append(("target", "target_verify", verify_start["timestamp_ns"], verify_end["timestamp_ns"], verify_end))
            proposed = verify_end["actual_tokens"]
            if proposed < 1:
                raise TraceError("target_verify must examine at least one proposal")
            if len(events) - cursor < proposed:
                raise TraceError("missing draft events")
            statuses = []
            for offset in range(1, proposed + 1):
                draft = events[cursor]
                cursor += 1
                if draft["kind"] != "draft" or draft["round"] != round_index or draft["offset"] != offset:
                    raise TraceError("draft events must be sequential and inside the round")
                statuses.append(draft["status"])
            if "rejected" in statuses:
                first_rejected = statuses.index("rejected")
                if statuses.count("rejected") != 1:
                    raise TraceError("a round may contain at most one rejected draft")
                if any(status != "unexamined" for status in statuses[first_rejected + 1:]):
                    raise TraceError("drafts after a rejection must be unexamined")
            elif "unexamined" in statuses:
                first_unexamined = statuses.index("unexamined")
                if any(status != "unexamined" for status in statuses[first_unexamined:]):
                    raise TraceError("accepted drafts may not follow unexamined drafts")
            accepted = statuses.count("accepted")
            rejected = statuses.count("rejected")
            unexamined = statuses.count("unexamined")
            if accepted + rejected + unexamined != proposed:
                raise TraceError("draft status counts do not sum to proposals")
            if rejected == 0 and unexamined:
                # Only an early accepted EOS can leave a proposed-but-unexamined suffix.
                pass

            if rejected:
                replay_start, replay_end, cursor = _phase_pair(events, cursor, "target",
                                                                "target_rejection_replay",
                                                                round_index, target_len)
                if replay_end["actual_tokens"] != accepted + 1:
                    raise TraceError("rejection replay must advance accepted plus one correction token")
                phase_pairs.append(("target", "target_rejection_replay", replay_start["timestamp_ns"], replay_end["timestamp_ns"], replay_end))
                target_after_work = replay_end["after_length"]
                correction = 1
            else:
                target_after_work = verify_end["after_length"]
                correction = 0

            bonus = 0
            if _expect_phase(events, cursor, "target_bonus_advance"):
                bonus_start, bonus_end, cursor = _phase_pair(events, cursor, "target",
                                                              "target_bonus_advance",
                                                              round_index, target_after_work)
                if bonus_end["actual_tokens"] != 1:
                    raise TraceError("bonus advance must move exactly one token")
                phase_pairs.append(("target", "target_bonus_advance", bonus_start["timestamp_ns"], bonus_end["timestamp_ns"], bonus_end))
                target_after_work = bonus_end["after_length"]
                bonus = 1

            shallow_replay_start, shallow_replay, cursor = _phase_pair(events, cursor, "shallow",
                                                     "shallow_committed_replay",
                                                     round_index, shallow_len)
            phase_pairs.append(("shallow", "shallow_committed_replay", shallow_replay_start["timestamp_ns"], shallow_replay["timestamp_ns"], shallow_replay))
            committed = shallow_replay["actual_tokens"]
            if committed < 1:
                raise TraceError("a D1/D2 round must commit at least one token")
            if cursor >= len(events) or events[cursor]["kind"] != "round_commit":
                raise TraceError("missing D1/D2 round_commit")
            commit = events[cursor]
            cursor += 1
            if commit["strategy"] != strategy or commit["round"] != round_index:
                raise TraceError("round_commit strategy/round mismatch")
            checks = {
                "proposed": proposed, "verified": accepted + rejected,
                "accepted": accepted, "rejected": rejected,
                "unexamined": unexamined, "correction": correction,
                "bonus": bonus, "committed": committed,
            }
            for name, expected in checks.items():
                if commit[name] != expected:
                    raise TraceError(f"round_commit {name} {commit[name]} != {expected}")
            expected_committed = accepted + correction + bonus
            if committed != expected_committed:
                raise TraceError("committed tokens do not equal accepted+correction+bonus")
            if target_after_work != target_len + committed:
                raise TraceError("target work length does not reconcile with committed tokens")
            total_proposed += proposed
            total_accepted += accepted
            total_rejected += rejected
            total_unexamined += unexamined
            total_correction += correction
            total_bonus += bonus
            for offset, status in enumerate(statuses, 1):
                row = per_offset.setdefault(offset, {"proposed": 0, "verified": 0,
                                                     "accepted": 0, "rejected": 0,
                                                     "unexamined": 0})
                row["proposed"] += 1
                row[status] += 1
                if status != "unexamined":
                    row["verified"] += 1

        if commit["target_before"] != target_len or commit["shallow_before"] != shallow_len:
            raise TraceError("round_commit before lengths disagree with the trace")
        if commit["output_before"] != output_len:
            raise TraceError("round_commit output_before disagrees with committed output")
        if commit["target_after"] != commit["target_before"] + commit["committed"]:
            raise TraceError("target cache length does not advance by committed tokens")
        if commit["shallow_after"] != commit["shallow_before"] + commit["committed"]:
            raise TraceError("shallow cache length does not advance by committed tokens")
        if commit["output_after"] != commit["output_before"] + commit["committed"]:
            raise TraceError("output length does not advance by committed tokens")
        if not (commit["target_after"] == commit["shallow_after"] == prompt + commit["output_after"]):
            raise TraceError("target/shallow cache lengths do not equal prompt plus output")
        target_len, shallow_len, output_len = (commit["target_after"],
                                               commit["shallow_after"],
                                               commit["output_after"])
        rounds = round_index
        if first_commit is None:
            first_commit = commit["timestamp_ns"]
            first_round_committed = commit["committed"]
        last_commit = commit["timestamp_ns"]
        if commit["eos_committed"]:
            if not eos_enabled:
                raise TraceError("eos_committed without an EOS token")
            eos_seen = True
        if output_len > budget:
            raise TraceError("committed output exceeds the token budget")

    if strategy != "D0" and output_len < budget and not eos_seen:
        raise TraceError("trace ends before budget exhaustion or EOS")
    if strategy == "D0" and output_len < budget and not eos_seen:
        raise TraceError("D0 trace ends before budget exhaustion or EOS")
    if strategy != "D0" and rounds == 0 and budget != 0:
        raise TraceError("nonempty budget must produce at least one round")
    if strategy != "D0" and total_unexamined and total_rejected == 0 and not eos_seen:
        raise TraceError("unexamined drafts require a rejection or committed EOS")

    if end["rounds"] != rounds or end["output_length"] != output_len:
        raise TraceError("run_end totals disagree with rounds")
    if end["target_length"] != target_len or end["shallow_length"] != shallow_len:
        raise TraceError("run_end cache lengths disagree with the last commit")
    if strategy == "D0":
        if end["proposed"] or end["accepted"] or end["rejected"]:
            raise TraceError("D0 run_end proposal counts must be zero")
    else:
        if end["proposed"] != total_proposed or end["accepted"] != total_accepted:
            raise TraceError("run_end proposal counts disagree with rounds")
        if end["rejected"] != total_rejected:
            raise TraceError("run_end rejection count disagrees with rounds")

    calls = {}
    occurrences = {}
    tokens = {}
    phase_seconds = {}
    prefill_end = events[0]["timestamp_ns"]
    for role, phase, start_ts, end_ts, event in phase_pairs:
        key = f"{role}:{phase}"
        calls[key] = calls.get(key, 0) + event["model_calls"]
        occurrences[key] = occurrences.get(key, 0) + 1
        tokens[key] = tokens.get(key, 0) + event["actual_tokens"]
        phase_seconds[key] = phase_seconds.get(key, 0.0) + (end_ts - start_ts) / 1e9
        if phase in PREFILL_PHASES:
            prefill_end = max(prefill_end, end_ts)

    def total(prefix):
        return sum(value for key, value in tokens.items() if key.startswith(prefix))

    target_tokens = total("target:")
    shallow_tokens = total("shallow:")
    target_prefill_tokens = tokens.get("target:target_prefill", 0)
    shallow_prefill_tokens = tokens.get("shallow:shallow_prefill", 0)
    target_decode_tokens = target_tokens - target_prefill_tokens
    start_ns = run["timestamp_ns"]
    end_ns = end["timestamp_ns"]
    total_seconds = (end_ns - start_ns) / 1e9
    prefill_seconds = (prefill_end - start_ns) / 1e9
    ttft = None if first_commit is None else (first_commit - start_ns) / 1e9
    # Decode excludes prefill but includes the complete first generation round.
    decode = (end_ns - prefill_end) / 1e9
    post_first = None if first_commit is None else (end_ns - first_commit) / 1e9
    committed = output_len
    tokens_after_first = (committed - first_round_committed
                          if first_commit is not None else None)
    offsets = []
    for offset in sorted(per_offset):
        row = dict(per_offset[offset])
        row["offset"] = offset
        row["acceptance_per_verified"] = (row["accepted"] / row["verified"]
                                          if row["verified"] else None)
        row["accepted_per_proposed"] = (row["accepted"] / row["proposed"]
                                       if row["proposed"] else None)
        offsets.append(row)

    return {
        "schema": SCHEMA,
        "strategy": strategy,
        "prompt_length": prompt,
        "max_new_tokens": budget,
        "draft_length": run["draft_length"],
        "greedy": run["greedy"],
        "temperature": run["temperature"],
        "eos_enabled": eos_enabled,
        "rounds": rounds,
        "committed_tokens": committed,
        "proposed_tokens": total_proposed,
        "verified_tokens": total_accepted + total_rejected,
        "accepted_tokens": total_accepted,
        "rejected_tokens": total_rejected,
        "unexamined_tokens": total_unexamined,
        "correction_tokens": total_correction,
        "bonus_tokens": total_bonus,
        "acceptance_per_verified": (total_accepted / (total_accepted + total_rejected)
                                    if total_accepted + total_rejected else None),
        "accepted_per_proposed": (total_accepted / total_proposed
                                  if total_proposed else None),
        "unexamined_per_proposed": (total_unexamined / total_proposed
                                    if total_proposed else None),
        "eos_committed": eos_seen,
        "per_offset": offsets,
        "work": {
            "target_calls": sum(value for key, value in calls.items() if key.startswith("target:")),
            "target_tokens": target_tokens,
            "target_prefill_calls": calls.get("target:target_prefill", 0),
            "target_prefill_tokens": target_prefill_tokens,
            "target_decode_tokens": target_decode_tokens,
            "shallow_calls": sum(value for key, value in calls.items() if key.startswith("shallow:")),
            "shallow_tokens": shallow_tokens,
            "shallow_prefill_calls": calls.get("shallow:shallow_prefill", 0),
            "shallow_prefill_tokens": shallow_prefill_tokens,
            "model_calls_by_phase": dict(sorted(calls.items())),
            "calls_by_phase": dict(sorted(calls.items())),
            "phase_occurrences_by_phase": dict(sorted(occurrences.items())),
            "target_phase_occurrences": sum(value for key, value in occurrences.items()
                                            if key.startswith("target:")),
            "shallow_phase_occurrences": sum(value for key, value in occurrences.items()
                                             if key.startswith("shallow:")),
            "tokens_by_phase": dict(sorted(tokens.items())),
            "target_tokens_per_committed_token": (target_decode_tokens / committed
                                                  if committed else None),
            "target_extra_tokens_over_committed": (target_decode_tokens - committed
                                                   if committed else None),
        },
        "timing": {
            "source": "fake_clock" if synthetic else "monotonic_host_clock",
            "synthetic": bool(synthetic),
            "total_seconds": total_seconds,
            "prefill_seconds": prefill_seconds,
            "ttft_seconds": ttft,
            "decode_seconds": decode,
            "post_first_commit_seconds": post_first,
            "first_round_committed_tokens": (first_round_committed
                                             if first_commit is not None else None),
            "tokens_after_first_commit": tokens_after_first,
            "committed_tokens_per_decode_second": (committed / decode
                                                   if committed and decode > 0 else None),
            "phase_seconds": dict(sorted(phase_seconds.items())),
        },
    }


def validate_trace(events, *, synthetic=False):
    """Validate and return the aggregate summary."""
    return summarize(events, synthetic=synthetic)


@dataclass(frozen=True)
class D0Result:
    tokens: tuple
    target: object
    rounds: int


def run_d0(target, prompt, max_new_tokens, *, eos=None, temperature=1.,
           greedy=False, rng=None, observer=None, clock=None):
    """Target-only baseline with the same prompt/budget/temperature/EOS rules.

    No shallow runner is used. Observer events use the same schema as D1/D2;
    shallow counters mirror committed output for schema reconciliation only and
    no shallow phase is emitted. Imports torch only when this helper is called.
    """
    if not prompt or type(max_new_tokens) is not int or max_new_tokens < 0:
        raise ValueError("nonempty prompt and nonnegative integer budget required")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("invalid temperature")
    from splot.speculative import distribution, draw
    import torch
    if rng is None:
        rng = torch.Generator().manual_seed(42)

    emit = None
    seq = 0
    if observer is not None:
        if not callable(observer):
            raise TypeError("observer must be callable")
        now = time.monotonic_ns if clock is None else clock

        def emit(kind, **fields):
            nonlocal seq
            stamp = now()
            if type(stamp) is not int:
                raise TypeError("clock must return integer nanoseconds")
            event = {"schema": SCHEMA, "seq": seq, "kind": kind,
                     "timestamp_ns": stamp}
            event.update(fields)
            observer(event)
            seq += 1

        emit("run_start", strategy="D0", draft_length=0,
             max_new_tokens=max_new_tokens, prompt_length=len(prompt),
             greedy=bool(greedy), temperature=float(temperature),
             eos=eos is not None)
        emit("phase_start", role="target", phase="target_prefill", round=0,
             before_length=0)
    state, _ = target.advance(None, list(prompt))
    if emit:
        emit("phase_end", role="target", phase="target_prefill", round=0,
             before_length=0, after_length=state.length,
             requested_tokens=len(prompt), actual_tokens=len(prompt),
             model_calls=1)

    output = []
    rounds = 0
    while len(output) < max_new_tokens:
        rounds += 1
        before = state.length
        if emit:
            emit("phase_start", role="target", phase="target_step",
                 round=rounds, before_length=before)
        p = distribution(state.next_logits, temperature)
        token = int(p.argmax()) if greedy else draw(p, rng)
        state, _ = target.advance(state, [token])
        output.append(token)
        if emit:
            emit("phase_end", role="target", phase="target_step",
                 round=rounds, before_length=before,
                 after_length=state.length, requested_tokens=1,
                 actual_tokens=1, model_calls=1)
            commit = {
                "schema": SCHEMA, "seq": seq, "kind": "round_commit",
                "timestamp_ns": now(),
                "round": rounds, "strategy": "D0", "proposed": 0,
                "verified": 0, "accepted": 0, "rejected": 0,
                "unexamined": 0, "correction": 0, "bonus": 0, "committed": 1,
                "target_before": before, "target_after": state.length,
                "shallow_before": before, "shallow_after": state.length,
                "output_before": len(output) - 1,
                "output_after": len(output),
                "eos_committed": bool(eos is not None and token == eos),
            }
            observer(commit)
            seq += 1
        if eos is not None and token == eos:
            break
    if emit:
        emit("run_end", rounds=rounds, proposed=0, accepted=0, rejected=0,
             output_length=len(output), target_length=state.length,
             shallow_length=state.length)
    return D0Result(tuple(output), state, rounds)


def synthetic_fixtures():
    """Deterministic fake-clock traces covering the task's required cases."""
    def build(strategy, prompt, budget, draft, eos=False):
        events = []
        stamp = [1_000_000_000]

        def add(kind, **fields):
            events.append({"schema": SCHEMA, "seq": len(events), "kind": kind,
                           "timestamp_ns": stamp[0], **fields})
            stamp[0] += 1_000_000

        add("run_start", strategy=strategy, draft_length=draft,
            max_new_tokens=budget, prompt_length=prompt, greedy=False,
            temperature=1.0, eos=eos)

        def phase(role, name, round_index, before, requested, actual, calls=None):
            if calls is None:
                calls = actual if name == "shallow_draft" else 1
            add("phase_start", role=role, phase=name, round=round_index,
                before_length=before)
            add("phase_end", role=role, phase=name, round=round_index,
                before_length=before, after_length=before + actual,
                requested_tokens=requested, actual_tokens=actual,
                model_calls=calls)
            return before + actual

        phase("target", "target_prefill", 0, 0, prompt, prompt)
        if strategy != "D0":
            phase("shallow", "shallow_prefill", 0, 0, prompt, prompt)
        return events, add, phase

    cases = {}

    events, add, phase = build("D1", 4, 3, 2)
    phase("shallow", "shallow_draft", 1, 4, 1, 1)
    phase("target", "target_verify", 1, 4, 2, 2)
    add("draft", round=1, offset=1, status="accepted", proposed=True, verified=True)
    add("draft", round=1, offset=2, status="accepted", proposed=True, verified=True)
    phase("target", "target_bonus_advance", 1, 6, 1, 1)
    phase("shallow", "shallow_committed_replay", 1, 4, 3, 3)
    add("round_commit", round=1, strategy="D1", proposed=2, verified=2,
        accepted=2, rejected=0, unexamined=0, correction=0, bonus=1,
        committed=3, target_before=4, target_after=7, shallow_before=4,
        shallow_after=7, output_before=0, output_after=3, eos_committed=False)
    add("run_end", rounds=1, proposed=2, accepted=2, rejected=0,
        output_length=3, target_length=7, shallow_length=7)
    cases["d1_all_accepted_bonus"] = events

    events, add, phase = build("D1", 4, 2, 2)
    phase("shallow", "shallow_draft", 1, 4, 1, 1)
    phase("target", "target_verify", 1, 4, 2, 2)
    add("draft", round=1, offset=1, status="rejected", proposed=True, verified=True)
    add("draft", round=1, offset=2, status="unexamined", proposed=True, verified=False)
    phase("target", "target_rejection_replay", 1, 4, 1, 1)
    phase("shallow", "shallow_committed_replay", 1, 4, 1, 1)
    add("round_commit", round=1, strategy="D1", proposed=2, verified=1,
        accepted=0, rejected=1, unexamined=1, correction=1, bonus=0,
        committed=1, target_before=4, target_after=5, shallow_before=4,
        shallow_after=5, output_before=0, output_after=1, eos_committed=False)
    phase("shallow", "shallow_draft", 2, 5, 0, 0)
    phase("target", "target_verify", 2, 5, 1, 1)
    add("draft", round=2, offset=1, status="accepted", proposed=True, verified=True)
    phase("shallow", "shallow_committed_replay", 2, 5, 1, 1)
    add("round_commit", round=2, strategy="D1", proposed=1, verified=1,
        accepted=1, rejected=0, unexamined=0, correction=0, bonus=0,
        committed=1, target_before=5, target_after=6, shallow_before=5,
        shallow_after=6, output_before=1, output_after=2, eos_committed=False)
    add("run_end", rounds=2, proposed=3, accepted=1, rejected=1,
        output_length=2, target_length=6, shallow_length=6)
    cases["d1_first_rejection_unexamined"] = events

    events, add, phase = build("D2", 4, 2, 2)
    phase("shallow", "shallow_draft", 1, 4, 0, 0)
    phase("target", "target_verify", 1, 4, 2, 2)
    add("draft", round=1, offset=1, status="accepted", proposed=True, verified=True)
    add("draft", round=1, offset=2, status="rejected", proposed=True, verified=True)
    phase("target", "target_rejection_replay", 1, 4, 2, 2)
    phase("shallow", "shallow_committed_replay", 1, 4, 2, 2)
    add("round_commit", round=1, strategy="D2", proposed=2, verified=2,
        accepted=1, rejected=1, unexamined=0, correction=1, bonus=0,
        committed=2, target_before=4, target_after=6, shallow_before=4,
        shallow_after=6, output_before=0, output_after=2, eos_committed=False)
    add("run_end", rounds=1, proposed=2, accepted=1, rejected=1,
        output_length=2, target_length=6, shallow_length=6)
    cases["d2_later_rejection"] = events

    events, add, phase = build("D1", 4, 5, 2, eos=True)
    phase("shallow", "shallow_draft", 1, 4, 1, 0)
    phase("target", "target_verify", 1, 4, 1, 1)
    add("draft", round=1, offset=1, status="accepted", proposed=True, verified=True)
    phase("shallow", "shallow_committed_replay", 1, 4, 1, 1)
    add("round_commit", round=1, strategy="D1", proposed=1, verified=1,
        accepted=1, rejected=0, unexamined=0, correction=0, bonus=0,
        committed=1, target_before=4, target_after=5, shallow_before=4,
        shallow_after=5, output_before=0, output_after=1, eos_committed=True)
    add("run_end", rounds=1, proposed=1, accepted=1, rejected=0,
        output_length=1, target_length=5, shallow_length=5)
    cases["d1_eos"] = events

    events, add, phase = build("D2", 4, 1, 2)
    phase("shallow", "shallow_draft", 1, 4, 0, 0)
    phase("target", "target_verify", 1, 4, 1, 1)
    add("draft", round=1, offset=1, status="accepted", proposed=True, verified=True)
    phase("shallow", "shallow_committed_replay", 1, 4, 1, 1)
    add("round_commit", round=1, strategy="D2", proposed=1, verified=1,
        accepted=1, rejected=0, unexamined=0, correction=0, bonus=0,
        committed=1, target_before=4, target_after=5, shallow_before=4,
        shallow_after=5, output_before=0, output_after=1, eos_committed=False)
    add("run_end", rounds=1, proposed=1, accepted=1, rejected=0,
        output_length=1, target_length=5, shallow_length=5)
    cases["d2_short_budget"] = events

    events, add, phase = build("D1", 4, 0, 2)
    add("run_end", rounds=0, proposed=0, accepted=0, rejected=0,
        output_length=0, target_length=4, shallow_length=4)
    cases["d1_zero_output"] = events

    events, add, phase = build("D0", 4, 2, 0)
    for round_index in (1, 2):
        before = 4 + round_index - 1
        phase("target", "target_step", round_index, before, 1, 1)
        add("round_commit", round=round_index, strategy="D0", proposed=0,
            verified=0, accepted=0, rejected=0, unexamined=0, correction=0,
            bonus=0, committed=1, target_before=before,
            target_after=before + 1, shallow_before=before,
            shallow_after=before + 1, output_before=round_index - 1,
            output_after=round_index, eos_committed=False)
    add("run_end", rounds=2, proposed=0, accepted=0, rejected=0,
        output_length=2, target_length=6, shallow_length=6)
    cases["d0_target_only"] = events
    events, add, phase = build("D1", 4, 4, 3)
    phase("shallow", "shallow_draft", 1, 4, 2, 2)
    phase("target", "target_verify", 1, 4, 3, 3)
    add("draft", round=1, offset=1, status="accepted", proposed=True, verified=True)
    add("draft", round=1, offset=2, status="accepted", proposed=True, verified=True)
    add("draft", round=1, offset=3, status="accepted", proposed=True, verified=True)
    phase("target", "target_bonus_advance", 1, 7, 1, 1)
    phase("shallow", "shallow_committed_replay", 1, 4, 4, 4)
    add("round_commit", round=1, strategy="D1", proposed=3, verified=3,
        accepted=3, rejected=0, unexamined=0, correction=0, bonus=1,
        committed=4, target_before=4, target_after=8, shallow_before=4,
        shallow_after=8, output_before=0, output_after=4, eos_committed=False)
    add("run_end", rounds=1, proposed=3, accepted=3, rejected=0,
        output_length=4, target_length=8, shallow_length=8)
    cases["d1_draft_three_calls"] = events

    events, add, phase = build("D2", 4, 3, 2)
    phase("shallow", "shallow_draft", 1, 4, 0, 0)
    phase("target", "target_verify", 1, 4, 2, 2)
    add("draft", round=1, offset=1, status="accepted", proposed=True, verified=True)
    add("draft", round=1, offset=2, status="rejected", proposed=True, verified=True)
    phase("target", "target_rejection_replay", 1, 4, 2, 2)
    phase("shallow", "shallow_committed_replay", 1, 4, 2, 2)
    add("round_commit", round=1, strategy="D2", proposed=2, verified=2,
        accepted=1, rejected=1, unexamined=0, correction=1, bonus=0,
        committed=2, target_before=4, target_after=6, shallow_before=4,
        shallow_after=6, output_before=0, output_after=2, eos_committed=False)
    phase("shallow", "shallow_draft", 2, 6, 0, 0)
    phase("target", "target_verify", 2, 6, 1, 1)
    add("draft", round=2, offset=1, status="accepted", proposed=True, verified=True)
    phase("shallow", "shallow_committed_replay", 2, 6, 1, 1)
    add("round_commit", round=2, strategy="D2", proposed=1, verified=1,
        accepted=1, rejected=0, unexamined=0, correction=0, bonus=0,
        committed=1, target_before=6, target_after=7, shallow_before=6,
        shallow_after=7, output_before=2, output_after=3, eos_committed=False)
    add("run_end", rounds=2, proposed=3, accepted=2, rejected=1,
        output_length=3, target_length=7, shallow_length=7)
    cases["d2_multiple_rounds"] = events

    return cases
