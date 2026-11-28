"""Reproducible T0019 attempt-002 candidate counts against same-vocab SPLOT.

Run after accounting.json exists:
    python candidate-counts.py > candidate-counts.json

NTP candidates carry no dormant MTP head. Optional head counts are reported
separately and are never used to claim NTP parameter parity.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def dense(vocab, width, layers, q_heads, kv_heads, head_dim, intermediate):
    q = q_heads * head_dim
    kv = kv_heads * head_dim
    if q_heads % kv_heads or head_dim % 2 or q != width:
        raise ValueError('require width=q_heads*head_dim and integer GQA grouping')
    params = vocab * width + layers * (width * (2 * q + 2 * kv)
                                       + 3 * width * intermediate + 2 * width) + width

    def decode(context):
        return layers * (width * (2 * q + 2 * kv) + 3 * width * intermediate
                         + 2 * width * context) + width * vocab

    def prefill(context):
        return context * (layers * (width * (2 * q + 2 * kv) + 3 * width * intermediate)
                          + width * vocab) + layers * q * context * (context + 1)

    def mtp_offsets(context, offsets=1):
        return offsets * context * (width * width + width * vocab)

    return {
        'config': {'vocab': vocab, 'width': width, 'layers': layers,
                   'q_heads': q_heads, 'kv_heads': kv_heads,
                   'head_dim': head_dim, 'intermediate': intermediate},
        'physical_ntp_no_aux': params,
        'optional_mtp_aux_params': width * width + width,
        'physical_with_optional_aux': params + width * width + width,
        'decode_macs': {str(t): decode(t) for t in (512, 2048)},
        'prefill_all_logits_macs': {str(t): prefill(t) for t in (512, 2048)},
        'optional_mtp_offset_macs': {str(t): mtp_offsets(t, 1) for t in (512, 2048)},
        'formula': {
            'params_ntp': 'V*d + L*(d*(2q+2kv)+3*d*m+2*d)+d',
            'optional_aux': 'd*d+d',
            'decode': 'L*(d*(2q+2kv)+3*d*m)+L*2*q*T+d*V',
            'prefill': 'T*(L*(d*(2q+2kv)+3*d*m)+d*V)+L*q*T*(T+1)',
        },
    }


def gaps(candidate, target):
    def gap(value, reference):
        return {'value': value, 'target': reference, 'relative_gap': value / reference - 1}
    ntp512, ntp2048 = target['ntp_512'], target['ntp_2048']
    return {
        'params_vs_ntp_base': gap(candidate['physical_ntp_no_aux'], ntp512['base']),
        'params_with_optional_aux_vs_splot_total': gap(
            candidate['physical_with_optional_aux'], ntp512['physical']),
        'prefill_512_vs_ntp': gap(candidate['prefill_all_logits_macs']['512'],
                                 ntp512['prefill_all_logits_macs']),
        'prefill_2048_vs_ntp': gap(candidate['prefill_all_logits_macs']['2048'],
                                   ntp2048['prefill_all_logits_macs']),
        'decode_512_vs_ntp': gap(candidate['decode_macs']['512'], ntp512['decode_macs']),
        'decode_2048_vs_ntp': gap(candidate['decode_macs']['2048'], ntp2048['decode_macs']),
    }


def main():
    accounting = json.loads((HERE / 'accounting.json').read_text(encoding='utf-8'))
    same_vocab = {row['vocab']: row for row in accounting['same_vocab']}
    candidates = {
        'v16384_iso_parameter_option': dense(16384, 512, 6, 8, 4, 64, 2432),
        'v16384_iso_forward_mac_option': dense(16384, 512, 6, 8, 2, 64, 2176),
        'v32768_iso_parameter_option': dense(32768, 512, 6, 8, 2, 64, 2176),
        'v32768_iso_forward_mac_option': dense(32768, 512, 6, 8, 2, 64, 1920),
    }
    for name, candidate in candidates.items():
        candidate['gaps'] = gaps(candidate, same_vocab[candidate['config']['vocab']])
    enumeration = [
        dense(16384, 512, 7, 8, 2, 64, 2048),
        dense(16384, 512, 6, 8, 4, 64, 2432),
        dense(16384, 512, 6, 8, 2, 64, 2176),
        dense(16384, 384, 11, 6, 2, 64, 2048),
        dense(32768, 512, 7, 8, 4, 64, 1920),
        dense(32768, 512, 6, 8, 4, 64, 1920),
        dense(32768, 512, 6, 8, 2, 64, 2176),
        dense(32768, 512, 6, 8, 2, 64, 1920),
    ]
    payload = {
        'schema': 'splot-t0019-candidate-counts-v2',
        'source_commit': '6e8911662ac56df665147cb79e412f0383cf71d8',
        'instruction_commit': 'a771c370efe662f17ddf8c06769cbd06a783388a',
        'status': 'corrected analytical inventory; no candidate selected, no training or performance result',
        'same_vocab_splot': {
            str(vocab): {
                'physical_with_aux': target['ntp_512']['physical'],
                'base_ntp': target['ntp_512']['base'],
                'auxiliary': target['ntp_512']['auxiliary'],
                'ntp_512': target['ntp_512'],
                'ntp_2048': target['ntp_2048'],
                'a2_512': target['a2_512'],
                'a2_2048': target['a2_2048'],
            } for vocab, target in same_vocab.items()
        },
        'prior_attempt_correction': {
            'prior_candidate_params': 41101312,
            'v16384_splot_total_with_aux': same_vocab[16384]['ntp_512']['physical'],
            'v16384_splot_base_ntp': same_vocab[16384]['ntp_512']['base'],
            'gap_vs_total_with_aux': 41101312 / same_vocab[16384]['ntp_512']['physical'] - 1,
            'gap_vs_ntp_base': 41101312 / same_vocab[16384]['ntp_512']['base'] - 1,
            'note': 'The prior -4.54% used the V32768 total; the same candidate is +11.80% versus the V16384 total with auxiliary and +12.26% versus the NTP base.',
        },
        'candidates': candidates,
        'enumeration': enumeration,
        'a2': {
            'status': 'deferred',
            'reason': 'SPLOT A2 has two NTP readouts and two offset readouts; a single dense offset head plus one readout is not a credible matched comparator.',
            'splot_v16384_a2_512': same_vocab[16384]['a2_512'],
            'splot_v16384_a2_2048': same_vocab[16384]['a2_2048'],
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
