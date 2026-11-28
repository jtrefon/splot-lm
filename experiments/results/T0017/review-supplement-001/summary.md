# T0017 attempt-002 review supplement 001 — tokenizer provenance, comparator chronology and checkpoint availability

Task T0017, attempt `002`, issue #77, PR #92. This supplement responds to the coordinator review of 2026-11-28 (review id `6098338804`, PR #92) and was authorized by the owner only for bounded, non-training evidence clarification.

**Scope:** no training, no package installation, no wheel download, no automatic retry, no source change, no original-evidence modification, no merge/closure and no T0020 launch. One bounded PyPI JSON metadata request was made to compare the cached wheel identity against the trusted release record; no wheel artifact was downloaded.

**Original evidence:** `experiments/results/T0017/attempt-002/` remains byte-identical. Its `result.json` SHA-256 is `78ff180087e797a2caec4b9aa4706e3046729bf7d754082955bec07bf2f86d29`, all 18 listed sibling hashes verify, and the exact REP validator still reports `STRUCTURALLY VALID: passed (not scientific acceptance)`. See `preservation.json` and `original-validator.txt`.

## 1. Tokenizer wheel provenance

### Trusted PyPI release record

The exact wheel matching the installed Linux x86_64 cp310-abi3 package is:

| Field | Value |
|---|---|
| Wheel filename | `tokenizers-0.23.3-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` |
| Official URL | `https://files.pythonhosted.org/packages/3a/40/6706b82693715581457c6d5423eaa7faae576bb0526c5738a57085eb4449/tokenizers-0.23.3-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl` |
| PyPI SHA-256 | `376851d22bcf9d650a5c3090bb83e6cf9e895fbf0595369fa4cd43c1f69b5f87` |
| PyPI MD5 | `afe505a64e67528eb936d0f58c2db256` |
| PyPI BLAKE2b-256 | `3a406706b82693715581457c6d5423eaa7faae576bb0526c5738a57085eb4449` |
| Wheel size | 3,396,193 B |
| Upload time | `2026-10-09T10:01:05.480462Z` |
| Requires-Python | `>=3.10` |

Source: `https://pypi.org/pypi/tokenizers/0.23.3/json`, fetched once at `2026-11-28T14:27:44Z`; response 31,436 B, SHA-256 `9480313d3d3c6cd7cbbcf5b4fe31eb7d92fb53efe8146c5aadaa2b8e46f38468`. The complete response is included as `pypi-tokenizers-0.23.3.json`.

### Local uv cache identity

The local cache entry is `wheels-v6/pypi/tokenizers/0.23.3-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64`, a symlink to the extracted wheel archive `archive-v0/OzOLxwZHZ7oAdsMx`. It predates attempt-002 (cache entry mtime `2026-10-14T11:17:59Z`) and was used offline; no wheel was downloaded during the attempt.

| Cache artifact | Observation |
|---|---|
| `.http` metadata, 641 B, SHA-256 `624e40c01b71cf429f504d8feedc3db63e800693ae8bcf7ff0283ca0498601ef` | Contains the exact wheel filename and URL above; ETag `afe505a64e67528eb936d0f58c2db256`, which equals the PyPI MD5 |
| `.msgpack` metadata, 995 B, SHA-256 `75b918d0c583b9bb68179225a5027a62c4d0bb78082d077e0a937aa19e8d8d25` | Package version 0.23.3 and dependency metadata |
| uv Simple API cache, 1,652,200 B, SHA-256 `1a2f415895b49bdea9df1ecef6ee0587e5d414097cb930895a92fb035aec4284` | Contains the same wheel filename and PyPI SHA-256 `376851d22bcf9d650a5c3090bb83e6cf9e895fbf0595369fa4cd43c1f69b5f87` |

### Installed package verification

The isolated venv has Python 3.12.7, `tokenizers==0.23.3`, wheel tag `cp310-abi3-manylinux_2_17_x86_64`, and `tokenizers.abi3.so` SHA-256 `4b1458a379296762a034d3bc89299c0c35b4eefb0577acca922f481b262c65a1`. Every one of the 32 SHA-256 entries in the installed wheel `RECORD` verified against the installed bytes (0 missing, 0 mismatched). `RECORD` SHA-256 is `594502c82016a506018a24f94c695aedbb52432e63e705b29c75e80e886500a5`; `WHEEL` and `METADATA` hashes are in `wheel-identity.json`.

Local tokenizer test command and result (exit 0): import `tokenizers`, `Tokenizer.from_file` on the verified local `tokenizer.json`, then encode/decode a fixed ASCII test string. Version 0.23.3; tokenizer JSON SHA-256 `104d3927bb55cdd2a911dbe21c8e755264756cedb5663bd39365c3d4cba3864b`; 19 encoded tokens; round-trip true. Raw corpus text and token arrays were not logged. See `tokenizer-test.log`.

### Honest limitation

The original `.whl` container is **not retained** by uv; only the extracted archive and HTTP metadata remain. Therefore the PyPI SHA-256 was **not recomputed from local wheel bytes**, and the installed file hashes alone do not prove the wheel container identity. The identity is recovered and corroborated by exact filename, official URL, wheel tag, PyPI MD5 equal to the cached ETag, the cached PyPI Simple index SHA-256, and the 32/32 installed `RECORD` match. The trusted release identity is the PyPI SHA-256 above.

### Owner permission basis

The prior owner authorization was exactly:

> I authorize T0017 attempt-002 on the RTX 4090 within the reviewed continuation specification and resource limits.

That authorization named the continuation and its resource limits but did **not** separately name the tokenizer wheel. The continuation requires “explicit local owner consent for wheel download/install.” The operator interpreted the authorization of the continuation—which includes the optional isolated 0.23.3 wheel path and caps—as the required consent. No separate wheel-specific approval sentence exists in the record. This is recorded for explicit maintainer decision; this supplement performs no new wheel download or installation. The current owner authorization covers only bounded non-training evidence clarification.

## 2. Initial resume comparator rejection and correction

This was a failed preflight of a required gate, not a pass. It was corrected before the official gate, and the original attempt-002 evidence did not record it; this supplement records it as a procedural deviation without altering the original hashes.

| Order | Action | Command / result | Exit |
|---:|---|---|---:|
| 1 | Selected wrong config | `configs/experiments/splot-nano-synthetic-fp8-muon.json` (SHA-256 `5171e50c…141160`) plus only `training.fp8_weight_reuse=true` (temp SHA-256 `2aa099ed…ceb7e2`) | — |
| 2 | Continuous preflight | 4 steps, CUDA; 43,055,076 parameters; run JSON SHA-256 `a429e337…0fd97b`; checkpoint 395,174,563 B, SHA-256 `e10181ce…d4cf6e` | 0 |
| 3 | Split preflight, part 1 | 2 steps, CUDA; same config; 43,055,076 parameters | 0 |
| 4 | Split preflight, part 2 | resume to 4 steps, CUDA; run JSON SHA-256 `273e0272…aa0c66`; checkpoint 395,175,715 B, SHA-256 `840dfdb5…c6f841` | 0 |
| 5 | Comparator | `python3 tools/rep/compare_resume.py <local-work>/gate/continuous <local-work>/gate/split --steps 4 --parameters 179952`; stderr `FAILED: unexpected inventory or step count`; cause: 43,055,076 ≠ preregistered 179,952 parameters | **1** |
| 6 | Corrected config | `configs/experiments/splot-smoke-synthetic-fp8-muon.json` (SHA-256 `370fb892…9642be`) plus only `training.fp8_weight_reuse=true` (temp SHA-256 `7c9e604b…a6dfd5`), same file both runs | — |
| 7 | Required gate | fresh continuous 4 steps and 2+2 resume; comparator with `--parameters 179952` | 0 (passed exactly; see original `resume-gate.json` / `resume-compare.json`) |

The corrective rerun was **not separately approved in a new owner message**. It remained within the owner-authorized attempt-002 and used the exact preregistered smoke configuration. No gate threshold, tolerance, step count or comparator expectation was changed; the wrong-config preflight was discarded and never relabeled as a pass.

Sanitized copies of the original preflight logs are included as `initial-nano-continuous.log`, `initial-nano-split-first.log`, `initial-nano-split-resume.log` and `initial-compare.stderr`, with raw and sanitized SHA-256 values in `resume-comparator-chronology.json`. Sanitization replaced only local paths with placeholders; commands, exit statuses, inventories and measurements are unchanged.

## 3. Checkpoint continued local availability

| Field | Observation |
|---|---|
| Relative name | `real-pilot/checkpoint.pt` |
| Status | Available now; regular file, not a symlink |
| Size | 395,220,707 B — matches published value |
| SHA-256 | `d011336f23ee82e6a56b594b44343fa4e4232d2cf70c7a0ae41112db400683b9` — matches published value |
| mtime UTC | `2026-11-28T14:04:54Z` |
| Filesystem | `/tmp`, type `tmpfs`, volatile |
| Durable local copy | **None found** under the owner home filesystem by exact size |
| Public upload | None |
| Checkpoint keys | `history`, `identity`, `model`, `optimizer`, `rng` |
| Identity / history | commit `d3752ee…50549`; 64 history steps |
| Model / optimizer | 43,055,076 elements across 198 tensors; optimizer 280 tensors; all finite |
| Weight-cache state | Absent |
| Load time | 0.153 s (`torch.load`, CPU, `weights_only=True`) |

The checkpoint is accessible at the local run directory, but it resides on volatile `/tmp` and no durable copy was found. If the tmpfs is cleared, only the published hashes, run JSON and evidence remain. No copy or upload was made in this supplement because the authorization was limited to verification/clarification.

## 4. Preservation and validation

- Original attempt-002 evidence: unchanged; `result.json` SHA-256 `78ff180087e797a2caec4b9aa4706e3046729bf7d754082955bec07bf2f86d29`; all 18 sibling hashes verify.
- Exact validator on original evidence: `STRUCTURALLY VALID: passed (not scientific acceptance)`.
- Supplement is a separate directory; no original file or hash was edited.
- No raw corpus, token arrays, tokenizer JSON, shards, tensors or public checkpoint upload is included.
- Supplement file hashes are in `files.json`.

## 5. Open items requiring maintainer decision

1. **Wheel container identity:** exact filename/URL/tag/PyPI SHA-256/MD5/ETag are recovered, but the original `.whl` bytes are not retained, so the SHA-256 cannot be independently recomputed locally. The published PyPI SHA-256 is the trusted release identity; no wheel was downloaded or installed now.
2. **Wheel-specific owner consent:** the prior authorization named attempt-002 within the continuation, not the wheel separately. Whether that satisfies the continuation’s “explicit local owner consent for wheel download/install” is a maintainer decision; this supplement does not claim it does.
3. **Comparator process deviation:** one failed wrong-config preflight occurred before the required gate. It was corrected and the corrected gate passed, but the original attempt-002 `deviations` list is empty; this supplement is the deviation record.
4. **Checkpoint durability:** the trained checkpoint is currently available and hash-verified but only on volatile tmpfs, with no durable copy.

T0017 remains open. This supplement does not change the original scientific result, does not claim a new pass for the failed preflight, and does not authorize T0020 or further compute.
