# REP v0.1 protocol tooling validation
2026-11-27 | Local CPU | No SPLOT GPU execution.
Executed python test_validate_result.py with stdlib Python, 9/9 passed: pass/failure/blocked, invalid status, hash mismatch, pin mismatch, missing check, unlisted file, deviation rejects pass. Runner 0.022s, not benchmark.
Executed compare_resume.py compare() against two self-produced tiny toy checkpoint/run pairs: identical tensors and non-timing fields accepted, deliberately altered model tensor rejected at model.x. This is toy evidence only, not actual SPLOT checkpoint comparison.
Local Git blob hashes (verified uploaded): validate_result.py c9cd74152850486af3f7f8653fcdcf819990d581; test_validate_result.py eed0735610112b1b9878fc72da41146a24d1d61c; compare_resume.py d7fb40f93be8a46f3d98f7c971cbb38b7cf878a8.
No GPU or full T0001 commands executed here. Schema validator is structural only; no authenticity or absolute agent adherence guarantee. Owner review and actual Linux task remain pending.
