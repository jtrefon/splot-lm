
import json
from pathlib import Path
E=Path(__file__).resolve().parent
a=json.loads((E/"accounting.json").read_text())
c=json.loads((E/"candidate-counts.json").read_text())
rows={r["name"]:r["stdout_json"] for r in a["commands"]}
assert rows["nano_a2_context512"]["physical"]==43055076
assert rows["nano_ntp_context512"]["physical"]==43055076
assert rows["nano_a2_context512"]["logical_blocks"]==14
assert rows["nano_ntp_context512"]["logical_blocks"]==12
assert c["candidates"]["existing_dense_8block_v32768"]["physical_params"]==25172352
p=c["candidates"]["v16384_iso_parameter"]; m=c["candidates"]["v16384_iso_forward_mac"]
assert abs(p["gaps_vs_ntp"]["physical_params"]["relative_gap"])<.05
assert abs(m["gaps_vs_ntp"]["prefill_all_logits_macs"]["512"]["relative_gap"])<.05
assert abs(m["gaps_vs_ntp"]["prefill_all_logits_macs"]["2048"]["relative_gap"])<.05
p32=c["candidates"]["v32768_iso_parameter"]; m32=c["candidates"]["v32768_iso_forward_mac"]
assert abs(p32["gaps_vs_ntp"]["physical_params"]["relative_gap"])<.05
assert abs(m32["gaps_vs_ntp"]["prefill_all_logits_macs"]["512"]["relative_gap"])<.05
assert abs(m32["gaps_vs_ntp"]["prefill_all_logits_macs"]["2048"]["relative_gap"])<.05
print("consistency OK: Nano 43055076; dense8 25172352; selected param/MAC gaps within declared windows")
