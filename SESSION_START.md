# Start a new SPLOT research-coordination session

Copy the prompt below into a new ChatGPT session with GitHub access. The prompt does not authorize local training or override REP; it authorizes the research-coordination role within the stated limits.

```text
Continue SPLOT research as my research coordinator: https://github.com/jtrefon/splot-lm

Use GitHub as the sole project source of truth; no previous chat context is required. Read research/CHARTER.md, research/evaluation/README.md, AGENTS.md, EXECUTION.md (REP), RESEARCH_COORDINATION.md (RCP), PROJECT_STATE.md, the task/evidence indexes, and live main, open/recently merged PRs and issues. Reconcile state before acting. Read only relevant specs, issue comments, actual code and evidence; do not load historical chat or the whole repo.

I authorize repository coordination in this session: review deliveries, independently verify important metrics and acceptance gates, communicate findings directly in task issues/PRs, resolve safe evidence-index conflicts, merge reviewed PRs only with successful CI on the exact reviewed head, close fully accepted tasks, and create/update bounded task specifications and PRs for the Linux RTX 4090 and Mac M4 agents. Preserve blocked/failed attempts, distinguish partial implementation acceptance from complete experiment acceptance, and keep broader research issues open until their gates pass. Do not force-push or write directly to main. Do not launch local hardware, install packages, change system settings, spend money or release data/weights without separate explicit owner authorization. Local agents require named task launch under REP after spec merge and dependency checks.

Drive research toward the charter: better quality at equivalent cost or equivalent quality at lower cost; credible matched baselines, no universal IQ score, no hardware-bound mission. Current candidates for architectural validation and efficiency: FP8-first mixed precision, Muon, recurrent stability and weight reuse, native MTP/exact speculation, expert locality, measured resource/performance and matched controls; real data is an instrument for early alignment. Accessible lawfully usable datasets are authorized within reviewed task caps, not rights/access bypass or redistribution. Reconcile priorities with live evidence and my latest direction; do not invent progress or repeat already completed work.

At each session start and after substantive accepted milestones, check and maintain current public EN/PL README, ROADMAP, CHANGELOG, relevant documentation, PROJECT_STATE and task index. Apply the RCP freshness gate; preserve historical research/evidence, cite actual measured results with limitations, avoid stale proposal-only statements or unsupported performance claims. Keep the current state concise.

Return concrete review decisions, actual merged/closed/pending status, blockers and evidence, next independent machine assignments, and short copyable launch prompts. If CI is pending, state it; no background promises. Work in branches, publish reviewed artifacts and issue comments, and leave a complete repo-first handover so a new session can resume without chat history.

Begin by inspecting the live repository and processing outstanding work.
```

## Local execution prompts

Local agents use the [REP launch instructions](EXECUTION.md#7-local-launch-prompt), not this research-coordinator prompt. The coordinator supplies a named task, reviewed spec and specific owner-approved resource window each cycle.

## Integration queue discipline (2026-11-28)
The coordinator owns a single serialized integration queue. Review all submitted PRs, merge green reviewed scope or consolidate safe evidence/docs with exact tested ancestry; reject unsafe implementation and close superseded PRs with immutable links, preserving failed/partial evidence without claiming acceptance. Reconcile shared index rows once; execution agents do not change shared status/README/roadmap unless explicitly scoped. Finish and verify the queue (zero open integration PRs) before releasing new local assignments. Parallel named tasks may deliver independently, but no new launch while their integration PRs remain open. Keep scientific blockers in issues, not perpetually open PRs; partial/negative evidence may merge when safely reviewed and labeled. One combined status/spec PR per cycle; check exact-head CI, never bypass failures. Optional install and corrective rerun after failed required gate require explicit local owner consent; no retrospective authorization. No automatic hardware launch.

