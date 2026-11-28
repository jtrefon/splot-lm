# T0017 attempt-002 review supplement 001 — post-review addendum

2026-11-28. Responds to the coordinator review of PR #92 at 2026-11-28T15:03Z. This addendum records later authorized actions without rewriting the earlier supplement files, which remain byte-identical.

## 1. Five missing supplement files corrected

The five files below were present in the local supplement directory with the exact bytes already listed in `files.json`, but were excluded from commit `38c28c950f40ed1dd10b792f9be2f05c81ed8988` by the repository’s `*.log` ignore rule. They were force-added unchanged; no content, command, measurement or sanitization was altered.

| File | Bytes | SHA-256 (matches original `files.json`) |
|---|---:|---|
| `execution.log` | 7,271 | `f7abc6fcbd71f6c6abbff9aa614f95e5e3c4d80248d88324ee2b29055e0d5b65` |
| `initial-nano-continuous.log` | 734 | `757ef9c78495992456dcdabfcd459c36c4cb52be5ae27fc7e80f45b8ce27927b` |
| `initial-nano-split-first.log` | 730 | `f311d7a28b326c63e14ca1e612b8488cc32b5265b2d37358906fee91890a2ae3` |
| `initial-nano-split-resume.log` | 744 | `016390a619df7675d7177be267d82a22e05542dee51e7ef00e5b0216def0f6ca` |
| `tokenizer-test.log` | 339 | `76482deea5837ea6c9a20c134081de49e82fdf04709e4c1caf37644a24f0229b` |

`files.json` was refreshed to the complete supplement inventory (all files except `files.json` itself). All listed hashes verify.

## 2. Durable private checkpoint copy

The owner separately authorized copying the existing trained checkpoint to durable private local storage. The copy was made to a new private local directory on the ext4 owner filesystem; the local path is withheld from public evidence.

| Check | Observation |
|---|---|
| Source preserved | Yes; source file still present |
| Source bytes / SHA-256 | 395,220,707 / `d011336f23ee82e6a56b594b44343fa4e4232d2cf70c7a0ae41112db400683b9` |
| Destination bytes / SHA-256 | 395,220,707 / `d011336f23ee82e6a56b594b44343fa4e4232d2cf70c7a0ae41112db400683b9` |
| Exact published size/hash | Yes |
| Filesystem | ext4, durable private local storage |
| Permissions | directory `0700`, file `0600` |
| Overwrite / delete / upload | None |
| Source-destination equality | Exact byte size and SHA-256 match |

The durable copy is not uploaded and its path is not published. `checkpoint-durability.json` is the machine-readable record. This supersedes the earlier “no durable copy” status in `checkpoint-availability.json` and `summary.md`; those earlier files remain unchanged to preserve provenance.

## 3. PR reconciliation

PR #92 was reconciled against current `main` `f49af5bfb0085cbbc7161d5b3354b8e8fe309696` (merge commit reported in the PR). The evidence index preserves both T0018 evidence rows and both T0017 evidence rows:

- T0017 attempt-001 (blocked qualification, preserved)
- T0017 attempt-002 (real pilot, passed)
- T0018 attempt-001 (observer evidence)
- T0018 attempt-002 (corrected observer evidence)

No evidence was deleted or overwritten. Hosted CPU CI was run on the combined final head; the exact run is reported in the PR and issue comment rather than embedded here to avoid self-referential hash churn.

## 4. Remaining blockers

1. Wheel-specific owner consent was not separately obtained; the earlier authorization and the operator interpretation remain an explicit deviation for maintainer disposition. No retrospective authorization is claimed.
2. Separate owner approval of the wrong-config corrective rerun was not obtained; the procedural deviation remains recorded and the first preflight remains a failure, not a pass.
3. The original tokenizer `.whl` container is absent, so the PyPI SHA-256 is the trusted release identity but was not locally recomputed from wheel bytes.
4. T0017 #77 remains open pending explicit maintainer disposition of these deviations; T0020 remains gated by its own reviewed amendment and separate owner launch.

No training, package installation, network download, automatic retry, merge or closure was performed in this addendum.
