"""Reproducible stdlib calculation for T0019 dense candidate counts.

Run from this directory after accounting.json exists:
    python candidate-counts.py > candidate-counts.json

Only integer parameter counts and analytical contraction MACs are computed.
Forward MACs are not training FLOPs, memory, latency or measured hardware work.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET_PHYSICAL = 43055076
TARGET_NTP_PREFILL = {512: 20602552320, 2048: 89657966592}
TARGET_NTP_DECODE = {512: 41416704, 2048: 48494592}
TARGET_A2_PREFILL = {512: 41893232640, 2048: 177236606976}
TARGET_A2_DECODE = {512: 83392512, 2048: 92829696}


def dense_counts(vocab, width, layers, q_heads, kv_heads, head_dim, intermediate, *, mtp=True):
    """Dense decoder GQA/RoPE/RMSNorm/SwiGLU, tied embeddings, no biases."""
    q = q_heads * head_dim
    kv = kv_heads * head_dim
    if q_heads % kv_heads or head_dim % 2 or q != width:
        raise ValueError("require width=q_heads*head_dim, integer GQA grouping and even head")
    embedding = vocab * width
    per_layer = width * (2 * q + 2 * kv) + 3 * width * intermediate + 2 * width
    final_norm = width
    params_no_mtp = embedding + layers * per_layer + final_norm
    mtp_params = width * width + width if mtp else 0
    params = params_no_mtp + mtp_params

    def decode(context):
        # One new token attending to `context` keys/values; one final readout.
        return layers * (width * (2 * q + 2 * kv) + 3 * width * intermediate
                         + 2 * width * context) + width * vocab

    def prefill(context):
        # All T causal outputs; attention sums 2*q*sum_{t=1..T}t = q*T*(T+1).
        linear = layers * (width * (2 * q + 2 * kv) + 3 * width * intermediate)
        return context * (linear + width * vocab) + layers * q * context * (context + 1)

    def mtp_offset_macs(context, offsets):
        # Auxiliary offset head: d*d projection + d*V output per token/offset.
        per_token = width * width + width * vocab
        return offsets * context * per_token

    return {
        "config": {
            "vocab": vocab, "width": width, "layers": layers,
            "q_heads": q_heads, "kv_heads": kv_heads, "head_dim": head_dim,
            "intermediate": intermediate, "tied_embeddings": True,
            "bias": False, "mtp_auxiliary_head": mtp,
        },
        "formula": {
            "physical_no_mtp": "V*d + L*(d*(2q+2kv) + 3*d*m + 2*d) + d",
            "mtp_aux": "d*d + d",
            "decode_macs": "L*(d*(2q+2kv)+3*d*m) + L*2*q*T + d*V",
            "prefill_macs": "T*(L*(d*(2q+2kv)+3*d*m) + d*V) + L*q*T*(T+1)",
            "mtp_offsets": "offsets*T*(d*d + d*V)",
        },
        "physical_no_mtp": params_no_mtp,
        "mtp_aux_params": mtp_params,
        "physical_params": params,
        "decode_macs": {str(t): decode(t) for t in (512, 2048)},
        "prefill_all_logits_macs": {str(t): prefill(t) for t in (512, 2048)},
        "mtp_offset_macs": {
            "1_offset": {str(t): mtp_offset_macs(t, 1) for t in (512, 2048)},
            "2_offsets": {str(t): mtp_offset_macs(t, 2) for t in (512, 2048)},
        },
    }


def gaps(candidate):
    result = {}
    for key in ("decode_macs", "prefill_all_logits_macs"):
        result[key] = {}
        target = TARGET_NTP_DECODE if key == "decode_macs" else TARGET_NTP_PREFILL
        for context in (512, 2048):
            value = candidate[key][str(context)]
            result[key][str(context)] = {
                "value": value,
                "target_ntp": target[context],
                "relative_gap": value / target[context] - 1,
            }
    result["physical_params"] = {
        "value": candidate["physical_params"],
        "target": TARGET_PHYSICAL,
        "relative_gap": candidate["physical_params"] / TARGET_PHYSICAL - 1,
    }
    return result


def main():
    accounting = json.loads((HERE / "accounting.json").read_text(encoding="utf-8"))
    splot = {row["name"]: row["stdout_json"] for row in accounting["commands"]}
    candidates = {
        "v16384_iso_parameter": dense_counts(16384, 512, 9, 8, 2, 64, 1920),
        "v16384_iso_forward_mac": dense_counts(16384, 512, 6, 8, 2, 64, 2816),
        "v32768_iso_parameter": dense_counts(32768, 512, 6, 8, 2, 64, 2176),
        "v32768_iso_forward_mac": dense_counts(32768, 512, 6, 8, 2, 64, 1920),
        "existing_dense_8block_v32768": dense_counts(32768, 384, 8, 6, 2, 64, 1024, mtp=False),
    }
    for candidate in candidates.values():
        candidate["gaps_vs_ntp"] = gaps(candidate)
    enumeration = [
        dense_counts(16384, 384, 14, 6, 2, 64, 1920),
        dense_counts(16384, 512, 9, 8, 2, 64, 2048),
        dense_counts(16384, 512, 11, 8, 2, 64, 1280),
        dense_counts(16384, 512, 6, 8, 2, 64, 2816),
        dense_counts(16384, 512, 7, 8, 2, 64, 2176),
        dense_counts(32768, 384, 11, 6, 2, 64, 2048),
        dense_counts(32768, 512, 8, 8, 2, 64, 1664),
        dense_counts(32768, 512, 6, 8, 2, 64, 1920),
        dense_counts(32768, 512, 6, 8, 2, 64, 2176),
    ]
    payload = {
        "schema": "splot-t0019-candidate-counts-v1",
        "source_commit": "6e8911662ac56df665147cb79e412f0383cf71d8",
        "instruction_commit": "f49af5bfb0085cbbc7161d5b3354b8e8fe309696",
        "status": "analytical only; no training, latency, memory or quality result",
        "targets": {
            "nano_physical": TARGET_PHYSICAL,
            "ntp_prefill_macs": {str(k): v for k, v in TARGET_NTP_PREFILL.items()},
            "ntp_decode_macs": {str(k): v for k, v in TARGET_NTP_DECODE.items()},
            "a2_prefill_macs": {str(k): v for k, v in TARGET_A2_PREFILL.items()},
            "a2_decode_macs": {str(k): v for k, v in TARGET_A2_DECODE.items()},
        },
        "splot_accounting": {
            name: {
                "physical": value["physical"],
                "base": value["base"],
                "auxiliary": value["auxiliary"],
                "logical_blocks": value["logical_blocks"],
                "decode_macs": value["decode_macs"],
                "prefill_all_logits_macs": value["prefill_all_logits_macs"],
                "cache_bytes": value["cache_bytes"],
            }
            for name, value in splot.items()
        },
        "candidates": candidates,
        "enumeration": enumeration,
        "selection": {
            "primary_vocab": 16384,
            "rationale": [
                "T0016 actual tokenizer vocabulary is 16384; V32768 is compatibility only.",
                "Width 512, head 64 and 8/2 GQA preserve integer head grouping and a conventional aspect ratio.",
                "V16384 iso-parameter is within 5% physical parameters; its forward MAC gap is explicit.",
                "V16384 iso-forward-MAC is within 5% prefill MAC at both 512 and 2048; its parameter gap is explicit.",
                "A2 auxiliary comparison requires a matching dense offset head, not an unmodified NTP-only DenseLM.",
            ],
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
