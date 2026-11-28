"""Consistency gate for T0019 attempt-002 corrected claims."""
import json
from pathlib import Path

E = Path(__file__).resolve().parent
accounting = json.loads((E / 'accounting.json').read_text(encoding='utf-8'))
counts = json.loads((E / 'candidate-counts.json').read_text(encoding='utf-8'))
targets = {row['vocab']: row for row in accounting['same_vocab']}

v16 = targets[16384]
v32 = targets[32768]
assert v16['ntp_512']['physical'] == 36763620
assert v16['ntp_512']['base'] == 36615780
assert v16['ntp_512']['auxiliary'] == 147840
assert v16['ntp_512']['prefill_all_logits_macs'] == 17381326848
assert v16['ntp_2048']['prefill_all_logits_macs'] == 76773064704
assert v32['ntp_512']['physical'] == 43055076
assert v32['ntp_512']['base'] == 42907236
assert v32['ntp_512']['prefill_all_logits_macs'] == 20602552320

prior = counts['prior_attempt_correction']
assert prior['prior_candidate_params'] == 41101312
assert abs(prior['gap_vs_total_with_aux'] - 0.1180) < 0.0001
assert abs(prior['gap_vs_ntp_base'] - 0.1225) < 0.0001

for name, candidate in counts['candidates'].items():
    vocab = candidate['config']['vocab']
    gap = candidate['gaps']['params_vs_ntp_base']
    assert gap['target'] == targets[vocab]['ntp_512']['base']
    assert candidate['gaps']['prefill_512_vs_ntp']['target'] == targets[vocab]['ntp_512']['prefill_all_logits_macs']
    assert candidate['gaps']['prefill_2048_vs_ntp']['target'] == targets[vocab]['ntp_2048']['prefill_all_logits_macs']
    assert abs(gap['relative_gap']) <= 0.13
    print(name, 'params', candidate['physical_ntp_no_aux'], 'param_gap', round(gap['relative_gap'], 6),
          'prefill512_gap', round(candidate['gaps']['prefill_512_vs_ntp']['relative_gap'], 6))

assert counts['a2']['status'] == 'deferred'
assert abs(counts['candidates']['v16384_iso_parameter_option']['gaps']['prefill_512_vs_ntp']['relative_gap'] - 0.092746) < 0.0001
assert abs(counts['candidates']['v16384_iso_forward_mac_option']['gaps']['prefill_512_vs_ntp']['relative_gap']) < 0.001
print('consistency OK: same-vocab targets, prior +11.80% correction, NTP candidate gaps, A2 deferred')
