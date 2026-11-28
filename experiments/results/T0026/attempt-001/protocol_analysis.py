"""T0026 prospective protocol analysis; stdlib only, no training or data access.

Reads accepted evidence/configs on the reviewed checkout, derives planning
tables, and writes analysis.json plus proposed-protocol.json. All compute and
wall-time figures are analytical estimates with explicit assumptions; they are
not measured training results.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
TASK_COMMIT = "003cf8db6126f0aaefdd599ac159f2503457f709"
SOURCE_COMMIT = TASK_COMMIT

sys.path.insert(0, str(ROOT / "tools"))
from accounting import DenseShape, Shape, account, dense_account  # noqa: E402


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def config_entry(name: str) -> dict:
    path = ROOT / "configs/experiments" / name
    payload = load_json(path)
    return {
        "file": name,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "model_family": payload.get("model_family", "splot"),
        "dense": payload["dense"],
        "training": payload["training"],
    }


s_cfg = config_entry("splot-nano-v16384-a0.json")
p_cfg = config_entry("dense-ntp-512x6-q8kv4-ffn2432.json")
c_cfg = config_entry("dense-ntp-512x6-q8kv2-ffn2176.json")

s_shape = Shape(d=384, vocab=16384, q_heads=6, kv_heads=2, head=64,
                gdn_heads=6, conv=4, dense=1024, expert=512, experts=8, topk=2)
s_acc = account(s_shape, loops=2, context=512, readouts=1, mtp=True, offsets=0)
p_acc = dense_account(DenseShape(vocab=16384, width=512, layers=6, q_heads=8,
                                 kv_heads=4, head=64, dense=2432), 512)
c_acc = dense_account(DenseShape(vocab=16384, width=512, layers=6, q_heads=8,
                                 kv_heads=2, head=64, dense=2176), 512)

assert s_acc["physical"] == 36_763_620, s_acc["physical"]
assert s_acc["base"] == 36_615_780, s_acc["base"]
assert s_acc["auxiliary"] == 147_840, s_acc["auxiliary"]
assert p_acc["physical"] == 35_527_168, p_acc["physical"]
assert c_acc["physical"] == 32_381_440, c_acc["physical"]

arms = {
    "S": {
        "label": "SPLOT A0 scan R2",
        "config": s_cfg,
        "physical_parameters": s_acc["physical"],
        "ntp_base_parameters": s_acc["base"],
        "dormant_mtp_parameters": s_acc["auxiliary"],
        "gradient_present_parameters_a0": s_acc["base"],
        "active_parameter_min": s_acc["target_unique_active_min"],
        "active_parameter_max": s_acc["target_unique_active_max"],
        "prefill_512_macs": s_acc["prefill_all_logits_macs"],
        "prefill_512_macs_per_token": s_acc["prefill_all_logits_macs"] / 512,
        "decode_512_macs": s_acc["decode_macs"],
        "cache_bytes": s_acc["cache_bytes"],
    },
    "P": {
        "label": "dense GQA parameter-matched",
        "config": p_cfg,
        "physical_parameters": p_acc["physical"],
        "ntp_base_parameters": p_acc["physical"],
        "dormant_mtp_parameters": 0,
        "gradient_present_parameters_a0": p_acc["physical"],
        "active_parameter_min": p_acc["physical"],
        "active_parameter_max": p_acc["physical"],
        "prefill_512_macs": p_acc["prefill_all_logits_macs"],
        "prefill_512_macs_per_token": p_acc["prefill_all_logits_macs"] / 512,
        "decode_512_macs": p_acc["decode_macs"],
        "cache_bytes": None,
    },
    "C": {
        "label": "dense GQA forward-MAC-matched",
        "config": c_cfg,
        "physical_parameters": c_acc["physical"],
        "ntp_base_parameters": c_acc["physical"],
        "dormant_mtp_parameters": 0,
        "gradient_present_parameters_a0": c_acc["physical"],
        "active_parameter_min": c_acc["physical"],
        "active_parameter_max": c_acc["physical"],
        "prefill_512_macs": c_acc["prefill_all_logits_macs"],
        "prefill_512_macs_per_token": c_acc["prefill_all_logits_macs"] / 512,
        "decode_512_macs": c_acc["decode_macs"],
        "cache_bytes": None,
    },
}

# Accepted T0016 aggregate ratios: bytes per content token, and materialized JSONL
# bytes per valid token. These are planning ratios, not a representativeness claim.
t0016_dataset = load_json(ROOT / "experiments/results/T0016/attempt-001/dataset-summary.json")
t0016_tokenizer = load_json(ROOT / "experiments/results/T0016/attempt-001/tokenizer-summary.json")
t0016_manifest = load_json(ROOT / "experiments/results/T0016/attempt-001/manifest.json")
content_bytes = sum(item["content_bytes"] for item in t0016_dataset["sources"])
raw_bytes = sum(item["raw_bytes"] for item in t0016_dataset["sources"])
full_content_tokens = t0016_tokenizer["roundtrip"]["content_tokens"]
valid_tokens = t0016_dataset["counts"]["overall"]["valid_tokens"]
dataset_bytes = (t0016_dataset["manifest_bytes"] + t0016_tokenizer["tokenizer_json_bytes"]
                 + sum(shard["bytes"] for shard in t0016_manifest["shards"]))
bytes_per_content_token = content_bytes / full_content_tokens
raw_bytes_per_content_token = raw_bytes / full_content_tokens
jsonl_bytes_per_valid_token = dataset_bytes / valid_tokens

# Corpus scenarios: unique valid training tokens; 5% validation and 5% test.
corpus_scenarios = []
for train_tokens in (1_000_000, 3_000_000, 5_000_000):
    val_tokens = ceil_div(train_tokens * 5, 100)
    test_tokens = val_tokens
    total_tokens = train_tokens + val_tokens + test_tokens
    corpus_scenarios.append({
        "name": f"{train_tokens // 1_000_000}M",
        "train_unique_valid_tokens": train_tokens,
        "validation_tokens": val_tokens,
        "test_tokens": test_tokens,
        "total_tokens": total_tokens,
        "train_blocks_512_ceil": ceil_div(train_tokens, 512),
        "validation_blocks_512_ceil": ceil_div(val_tokens, 512),
        "test_blocks_512_ceil": ceil_div(test_tokens, 512),
        "extracted_utf8_bytes_est": round(total_tokens * bytes_per_content_token),
        "raw_bytes_est": round(total_tokens * raw_bytes_per_content_token),
        "compact_16b_per_token_est": total_tokens * 16,
        "compact_13_to_19b_per_token_est": [total_tokens * 13, total_tokens * 19],
        "jsonl_current_ratio_est": round(total_tokens * jsonl_bytes_per_valid_token),
        "assumptions": [
            "train/validation/test tokens are valid content-token proxies",
            "3.905 UTF-8 bytes per content token and 4.870 raw bytes per content token from T0016",
            "16 B/token compact binary planning assumption (13-19 B range)",
            "48.26 JSONL bytes per valid token from the current T0016 artifact",
        ],
    })

# Analytical training compute. 3x forward MACs approximates backward; 2 FLOPs/MAC.
# Sustained throughput values are deliberately conservative small-model assumptions.
compute = []
for arm_key, arm in arms.items():
    for tokens in (1_000_000, 3_000_000, 5_000_000):
        fwd = arm["prefill_512_macs_per_token"] * tokens
        train_macs = 3 * fwd
        train_flops = 2 * train_macs
        compute.append({
            "arm": arm_key,
            "tokens": tokens,
            "forward_macs": round(fwd),
            "training_macs_est_3x_forward": round(train_macs),
            "training_flops_est_6x_forward_macs": round(train_flops),
            "gpu_seconds_at_5_tflops": round(train_flops / (5e12), 3),
            "gpu_seconds_at_10_tflops": round(train_flops / (10e12), 3),
            "gpu_seconds_at_20_tflops": round(train_flops / (20e12), 3),
            "assumptions": [
                "analytical forward contraction MACs only; embeddings/norms/scalar ops excluded",
                "backward approximated as 2x forward MACs (3x total)",
                "2 FLOPs per MAC",
                "5/10/20 TFLOP/s sustained small-model throughput, not a hardware peak or measured result",
            ],
        })

# Wall-time anchor from accepted T0017: 81.042418144876 s for 64 updates.
t0017_seconds = 81.042418144876
t0017_updates = 64
seconds_per_update = t0017_seconds / t0017_updates

budget_stages = []
for name, positions_per_run, runs, note in (
    ("S0 feasibility", 65_536, 9, "128 updates/run; matches the T0027 9-run envelope"),
    ("S1 learning curve", 196_608, 9, "384 updates/run; four validation milestones; ~73 min analytical, leaves overhead"),
    ("S2 confirmation", 1_048_576, 9, "2,048 updates/run; ~21% of a 5M-token corpus per run; multi-window"),
    ("S3 scaling", 5_242_880, 9, "10,240 updates/run; one 5M-token epoch per run; multi-window"),
):
    updates = ceil_div(positions_per_run, 512)
    seconds_per_run = updates * seconds_per_update
    total_seconds = seconds_per_run * runs
    budget_stages.append({
        "name": name,
        "positions_per_run": positions_per_run,
        "updates_per_run": updates,
        "runs": runs,
        "total_positions": positions_per_run * runs,
        "total_valid_tokens_est": round(positions_per_run * runs * (46_204 / 65_536)),
        "seconds_per_run_at_t0017_anchor": round(seconds_per_run, 3),
        "total_seconds_at_t0017_anchor": round(total_seconds, 3),
        "total_minutes_at_t0017_anchor": round(total_seconds / 60, 2),
        "note": note,
    })

# Paired-seed precision planning. t critical values are hard-coded for the
# listed degrees of freedom; MDE is an 80% power normal/t approximation, not a
# guarantee, and no variance is assumed beyond the labelled scenario.
precision_rows = []
tcrit = {3: 4.303, 5: 2.776, 8: 2.365, 12: 2.201, 20: 2.093}
tpower = {3: 1.886, 5: 1.533, 8: 1.415, 12: 1.363, 20: 1.328}
for n in (3, 5, 8, 12, 20):
    row = {"paired_seeds": n, "df": n - 1, "tcrit_975": tcrit[n],
           "tpower_80": tpower[n], "by_sd": {}}
    for sd in (0.02, 0.05, 0.10):
        half = tcrit[n] * sd / math.sqrt(n)
        mde = (tcrit[n] + tpower[n]) * sd / math.sqrt(n)
        row["by_sd"][f"{sd:.2f}"] = {
            "ci95_half_width_nats": round(half, 4),
            "mde80_approx_nats": round(mde, 4),
        }
    precision_rows.append(row)

normal_n = {}
for sd in (0.02, 0.05, 0.10):
    normal_n[f"{sd:.2f}"] = {}
    for margin in (0.05, 0.03):
        n = max(3, math.ceil(((1.96 + 0.8416) * sd / margin) ** 2))
        normal_n[f"{sd:.2f}"][f"{margin:.2f}"] = n

margins = {
    "status": "planning defaults only; not preregistered and not a claim",
    "primary_contrast": "S - C (compute-matched dense), token-weighted final validation NTP NLL",
    "secondary_contrast": "S - P (parameter-matched dense)",
    "noninferiority_nll_nats": 0.05,
    "superiority_nll_nats": -0.03,
    "noninferiority_bits_per_byte": 0.01,
    "capability_guardrail_pp": -2.0,
    "training_cost_guardrail_pct": 10.0,
    "decision_rule": (
        "Noninferiority only if the two-sided 95% paired CI upper bound is below "
        "+0.05 nats; superiority only if it is below -0.03 nats. No significance "
        "or equivalence claim is made from a pilot with inadequate seed/family count."
    ),
    "justification": (
        "0.05 nats is a small but non-trivial language-modeling difference and is "
        "the starting point used in T0019 planning; -0.03 nats is a stricter "
        "meaningful-advantage threshold. Both must be revised after the observed "
        "paired SD and family variance are available."
    ),
}

probes = {
    "status": "design only; no instances generated, no test access",
    "principles": [
        "frozen before any training run and never used for tuning",
        "generators, templates, entities and instances disjoint from train/validation",
        "chance and floor/ceiling reported; uninformative if all arms near chance",
        "held-out test probe set untouched until final evaluation",
        "no universal capability or intelligence claim",
    ],
    "families": [
        {
            "name": "key_value_retrieval",
            "chance": 1 / 16,
            "instances": 256,
            "scoring": "exact answer match",
            "floor_rule": "chance + 2 binomial SE",
            "ceiling_rule": "uninformative if every arm > 0.90",
            "contamination_audit": "13-gram overlap with train/validation < 0.1%",
        },
        {
            "name": "compositional_rule_completion",
            "chance": 1 / 8,
            "instances": 256,
            "scoring": "exact answer match on held-out rule",
            "floor_rule": "chance + 2 binomial SE",
            "ceiling_rule": "uninformative if every arm > 0.90",
            "contamination_audit": "held-out symbol mapping and templates",
        },
        {
            "name": "document_order_reconstruction",
            "chance": 0.5,
            "instances": 128,
            "scoring": "pairwise order accuracy",
            "floor_rule": "chance + 2 binomial SE",
            "ceiling_rule": "uninformative if every arm > 0.95",
            "contamination_audit": "synthetic sentence pairs, no corpus text",
        },
    ],
}

bpb = {
    "current_format": "splot-pilot-blocks-v1",
    "current_available": [
        "token IDs, document IDs, positions, valid mask and shifted labels",
        "source-level raw/content byte counts and tokenizer roundtrip totals",
        "shard token/valid/NTP/offset2 counts",
    ],
    "current_missing": [
        "per-token UTF-8 byte length and byte offset",
        "per-document original byte start/end and document hash",
        "per-token source family and language identifiers",
        "explicit BOS/EOS/PAD control mask",
        "tokenizer token-to-byte table in the evidence artifact",
        "shard-level byte coverage and per-family byte denominators",
    ],
    "required_schema": {
        "format": "splot-pilot-blocks-v2",
        "tokenizer": {
            "token_bytes": "length-vocab byte strings (hex/base64), specials flagged",
            "actual_vocab_size": 16384,
            "normalizer": None,
            "byte_level": True,
            "special_token_ids": [0, 1, 2, 3, 4, 5, 6, 7],
        },
        "block_fields": [
            "input_ids", "document_ids", "position_ids", "valid", "labels_1", "labels_2",
            "token_byte_lengths", "token_byte_offsets", "control_mask",
            "source_family_ids", "language_ids", "document_byte_start", "document_byte_end",
        ],
        "manifest_fields": [
            "tokenizer_sha256", "source_snapshot_sha256", "logical_sha256",
            "byte_coverage_sha256", "shards[].bytes", "shards[].content_bytes",
            "shards[].family_token_counts", "shards[].family_byte_counts",
            "shards[].language_token_counts", "shards[].language_byte_counts",
        ],
        "evaluation_rule": (
            "Primary BPB uses content-token labels only: numerator is the summed "
            "NLL of predicted content tokens; denominator is the summed UTF-8 byte "
            "length of those same target tokens. BOS/EOS/PAD NLL is reported "
            "separately and is never divided by zero bytes. Reversibility requires "
            "decode(token_bytes) == original UTF-8 bytes for every document."
        ),
    },
    "reconciliation": [
        "sum of token_byte_lengths per document equals document_byte_end - document_byte_start",
        "sum of document bytes per shard equals manifest content_bytes",
        "no gaps or overlaps in byte offsets",
        "ordinary NTP NLL and content-only BPB are reported separately",
        "EOS NLL per document is reported as a control metric",
    ],
}

gdn_localization = {
    "status": "separate diagnostic proposal only; no production change or equivalence claim",
    "trigger": "T0025 full-model chunk64 failed hidden/logit/offset2 tensor gates and top2 routing at T512",
    "steps": [
        "reproduce the exact T0025 initialization, input, partition and scan control",
        "compare scan vs chunk64 at FP32 recurrence with autocast disabled",
        "insert BF16 casts one at a time at projection input, residual input and router input",
        "record per-module max abs/relative L2, route logits, top1/top2 margins and rank flips",
        "group by depth (prelude/core/coda), pool (0-3), pass (C1/C2) and token position",
        "run fixed-route teacher forcing to separate routing divergence from contraction order",
    ],
    "gates": [
        "FP32 recurrence state/output within 1e-9 relative L2 and max abs",
        "no threshold relaxation from T0025",
        "report first nonzero top2 flip and per-group disagreement counts, not inferred causes",
        "teacher-forced fixed-route results are diagnostic only and cannot establish exact equivalence",
    ],
    "outputs": [
        "per-module numerical table", "route margin/rank table", "depth/pool flip table",
        "minimal reproduction script and sanitized log",
    ],
}

unresolved = [
    "corpus sources, rights evidence and family assignments",
    "whether 8 train / 4 validation / 4 test families are sufficient for cluster uncertainty at 1M tokens",
    "tokenizer fit sample size and whether one 16k tokenizer is frozen across all scales",
    "optimized dense comparator implementation and its acceptance tests",
    "final paired-seed count after observed paired SD",
    "final noninferiority/superiority margins after variance and family review",
    "available multi-window compute budget and whether 5M-token confirmation is feasible",
    "v1 -> v2 schema migration and independent byte-alignment audit",
    "whether conditional-memory vs expert-capacity design belongs in the first #86 follow-up",
]

dependencies = [
    {"id": "D1", "item": "T0026 design review", "depends_on": []},
    {"id": "D2", "item": "v2 byte-aligned dataset schema + independent audit", "depends_on": ["D1"]},
    {"id": "D3", "item": "optimized dense SDPA comparator implementation + CPU/CUDA correctness gates", "depends_on": ["D1"]},
    {"id": "D4", "item": "lawful corpus acquisition, rights registry and family split", "depends_on": ["D2"]},
    {"id": "D5", "item": "frozen tokenizer fit on train-only sample", "depends_on": ["D4"]},
    {"id": "D6", "item": "1M-token matched feasibility and paired-SD estimate", "depends_on": ["D2", "D3", "D4", "D5"]},
    {"id": "D7", "item": "5M-token confirmation with predeclared seeds/margins", "depends_on": ["D6"]},
    {"id": "D8", "item": "GDN chunk64 localization diagnostic", "depends_on": ["D6"]},
    {"id": "D9", "item": "conditional-memory vs expert-capacity design for #86", "depends_on": ["D7"]},
]

non_claims = [
    "no model quality, superiority, equivalence or scaling claim",
    "no empirical variance, power or family-generalization estimate from T0016",
    "no training, dataset acquisition, tokenizer fit or test access",
    "no optimized dense implementation or measured cost result",
    "no architecture adoption, pre-registration of final margins or task launch",
]

analysis = {
    "schema": "splot-t0026-analysis-v1",
    "task_id": "T0026",
    "task_commit": TASK_COMMIT,
    "source_commit": SOURCE_COMMIT,
    "configs": {"S": s_cfg["sha256"], "P": p_cfg["sha256"], "C": c_cfg["sha256"]},
    "arms": arms,
    "t0016_ratios": {
        "content_bytes": content_bytes,
        "raw_bytes": raw_bytes,
        "full_content_tokens": full_content_tokens,
        "valid_tokens": valid_tokens,
        "dataset_bytes": dataset_bytes,
        "utf8_bytes_per_content_token": round(bytes_per_content_token, 6),
        "raw_bytes_per_content_token": round(raw_bytes_per_content_token, 6),
        "jsonl_bytes_per_valid_token": round(jsonl_bytes_per_valid_token, 6),
    },
    "corpus_scenarios": corpus_scenarios,
    "compute_scenarios": compute,
    "wall_time_anchor": {
        "source": "T0017 attempt-002 real pilot, accepted feasibility",
        "seconds": t0017_seconds,
        "updates": t0017_updates,
        "seconds_per_update": round(seconds_per_update, 6),
        "caveat": "FP8 A2 43M model; a conservative overhead anchor, not a BF16 A0 prediction",
    },
    "budget_stages": budget_stages,
    "precision_planning": {
        "paired_seed_rows": precision_rows,
        "normal_approx_n_for_margin": normal_n,
        "caveat": "assumed paired SD; no observed variance exists; not a power guarantee",
    },
    "margins": margins,
    "probes": probes,
    "bpb": bpb,
    "gdn_localization": gdn_localization,
    "unresolved": unresolved,
    "dependencies": dependencies,
    "non_claims": non_claims,
}

(OUT / "analysis.json").write_text(
    json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

protocol = {
    "schema": "splot-t0026-proposed-protocol-v1",
    "status": "proposed; not preregistered; no training authorized",
    "task_id": "T0026",
    "task_commit": TASK_COMMIT,
    "source_commit": SOURCE_COMMIT,
    "primary_question": (
        "At matched valid-token and forward-contraction budgets, does SPLOT A0 "
        "scan R2 preserve or improve held-out token-weighted final NTP NLL versus "
        "a credible optimized dense GQA comparator without material capability "
        "regression, at comparable or lower training cost?"
    ),
    "arms": {
        key: {
            "label": value["label"],
            "config_file": value["config"]["file"],
            "config_sha256": value["config"]["sha256"],
            "model_family": value["config"]["model_family"],
            "physical_parameters": value["physical_parameters"],
            "ntp_base_parameters": value["ntp_base_parameters"],
            "dormant_mtp_parameters": value["dormant_mtp_parameters"],
            "active_parameter_range": [value["active_parameter_min"], value["active_parameter_max"]],
            "prefill_512_macs": value["prefill_512_macs"],
        } for key, value in arms.items()
    },
    "contrasts": {
        "primary": "S - C, paired by seed, final token-weighted validation NTP NLL",
        "secondary": "S - P, paired by seed; interpret with multiplicity caution",
        "cost": "analytical forward/training MACs and measured wall time; report achieved gaps",
    },
    "regimes": [
        {"name": "matched_token", "rule": "identical token stream, order, valid labels and update count"},
        {"name": "matched_forward_mac", "rule": "C within 0.1% of S prefill MACs/token at 512; primary cost regime"},
        {"name": "matched_parameter", "rule": "P within 3% of S NTP base parameters; secondary"},
        {"name": "matched_training_compute", "rule": "if used, adjust updates to equal estimated 3x-forward MACs; disclose achieved gap"},
    ],
    "endpoints": {
        "primary": "final token-weighted validation final NTP NLL (nats/label)",
        "secondary": [
            "content-only byte-aligned BPB (bits/byte)",
            "ordinary NTP NLL and EOS/control NLL separately",
            "per-language and per-family token-weighted NLL",
            "macro-average family NLL",
            "paired seed difference mean/median/min/max/SD/95% CI",
            "fixed-budget capability probes",
            "analytical training MACs, measured wall time, checkpoint bytes",
            "learning curves and censored time-to-target",
        ],
        "excluded": ["test metrics before freeze", "token-level bootstrap as family inference", "single-run significance claims"],
    },
    "margins": margins,
    "seeds": {
        "planning_minimum": 3,
        "confirmation_range": [5, 12],
        "default_pairing": [42, 43, 44],
        "rule": "same seed controls data order and RNG across arms; paired differences only",
        "final_count": "unresolved until observed paired SD and family variance",
    },
    "budgets": {
        "corpus_scenarios": corpus_scenarios,
        "budget_stages": budget_stages,
        "wall_time_anchor": analysis["wall_time_anchor"],
        "compute_scenarios": compute,
        "validation_cadence": "at fixed token milestones (quarter-budget preferred) and final; primary endpoint final checkpoint",
        "checkpoint_cadence": "every 16-32 updates with atomic write and identity hash; no early stopping for primary",
    },
    "precision_planning": analysis["precision_planning"],
    "corpus": {
        "planning_range_unique_train_tokens": [1_000_000, 5_000_000],
        "stages": ["1M descriptive/paired-SD", "3M intermediate", "5M confirmation"],
        "minimum_families": {"train": 8, "validation": 4, "test": 4},
        "preferred_families": {"train": 12, "validation": 6, "test": 6},
        "tokenizer": "one frozen byte-level BPE, actual vocab 16,384, train-only deterministic fit",
        "split_rule": "family-level disjoint; no document/family crosses splits; test frozen",
        "dedup": "exact + normalized + near-duplicate MinHash at document level; family-level dedup; 13-gram cross-split audit",
        "rights": "public-domain/CC0/CC-BY with per-source registry, revision, license and rights evidence; no acquisition in T0026",
    },
    "bpb": bpb,
    "probes": probes,
    "gdn_localization": gdn_localization,
    "unresolved": unresolved,
    "dependency_graph": dependencies,
    "non_claims": non_claims,
}
(OUT / "proposed-protocol.json").write_text(
    json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

# Source references hashed after reading; no content is copied.
source_paths = [
    "research/CHARTER.md", "research/evaluation/README.md",
    "research/evaluation/T0023-acceptance.md", "research/evaluation/T0025-disposition.md",
    "experiments/tasks/T0026-matched-evaluation-protocol.md",
    "experiments/tasks/T0027-matched-ntp-pilot.md",
    "experiments/results/T0016/attempt-001/summary.md",
    "experiments/results/T0016/attempt-001/dataset-summary.json",
    "experiments/results/T0016/attempt-001/tokenizer-summary.json",
    "experiments/results/T0016/attempt-001/manifest.json",
    "experiments/results/T0019/attempt-002/summary.md",
    "experiments/results/T0023/attempt-003/summary.md",
    "experiments/results/T0023/attempt-003/proposed-matrix.json",
    "experiments/results/T0025/attempt-001/summary.md",
    "configs/experiments/splot-nano-v16384-a0.json",
    "configs/experiments/dense-ntp-512x6-q8kv4-ffn2432.json",
    "configs/experiments/dense-ntp-512x6-q8kv2-ffn2176.json",
    "tools/accounting.py", "tools/experiment.py", "tools/pilot_ingestion.py",
    "src/splot/dense.py", "src/splot/hybrid.py", "src/splot/mtp.py",
]
refs = {}
for rel in source_paths:
    path = ROOT / rel
    refs[rel] = {"sha256": sha256(path), "bytes": path.stat().st_size}
(OUT / "source-refs.json").write_text(
    json.dumps({"schema": "splot-t0026-source-refs-v1", "source_commit": SOURCE_COMMIT,
                "refs": refs}, indent=2, sort_keys=True, allow_nan=False) + "\n",
    encoding="utf-8")

print("wrote analysis.json, proposed-protocol.json, source-refs.json")
print("arms:", {k: (v["physical_parameters"], v["prefill_512_macs"]) for k, v in arms.items()})
print("ratios:", round(bytes_per_content_token, 6), round(raw_bytes_per_content_token, 6),
      round(jsonl_bytes_per_valid_token, 6))
print("budget minutes:", [row["total_minutes_at_t0017_anchor"] for row in budget_stages])
