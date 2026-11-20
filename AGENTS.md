# Agent guide

This is Ouroboros, an experimental local model-adaptation toolkit. Read README.md's capability table and docs/AUDIT_AND_MIGRATION.md before making claims based on old class names or historical releases.

## Work safely

- Stay on the user's assigned working branch. Do not create hidden experiment branches, auto-merge generated code or rewrite published releases. Explain measured work and limitations before publishing it.
- Never fabricate benchmark gains, provenance, restored files, bridge execution or successful uploads.
- Core commands are offline-first. Cloning upstream source is not installation or permission to run autonomous upstream agents.
- Generated Python is a proposal, not trusted application code. Read SECURITY.md before introducing execution.
- Keep train/replay, validation and optional final-test data separate. Proposers must not receive held-out labels or access their files. Normalized deduplication is not semantic decontamination.
- Do not delete source trees, models, datasets, arbitrary checkpoint folders or system caches as a shortcut. Managed pruning must remain run-scoped and checksum verified.

## Setup / checks

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
python -m pytest -q -m 'not ml'
ruff check ouroboros tests scripts
ruff format --check ouroboros tests scripts
python scripts/check_docs.py
ouroboros run --config examples/local_sft.json --check-only
ouroboros weights trial --config examples/weight_trial.json --check-only
ouroboros demo --generations 3
```

`python scripts/install.py --profile core --dev` creates the local environment with optional MCP; `--profile llm` adds same-model inference/GGUF smoke provisioning, `--profile research` adds Math-Verify and the optional structured-command decoder, and `--profile train` adds PEFT. `ouroboros init NEW_DIRECTORY` packages its own tiny fixtures. Run `self-edit --config CONFIG --check-only` before a real run. Model downloads are explicit `models fetch` (Hub) / `models fetch-smoke` (pinned GitHub derivative), not side effects of inference.

Core dependencies include NumPy, PyYAML and safetensors, **not Torch**. `pip install -e '.[train]' -c constraints/ml-tested.txt` enables CPU Torch/direct-edit and tiny random local-LM tests without pretrained downloads. No GPU is assumed. Run the full `pytest` suite in that environment; mark new Torch-dependent tests `ml`. Check-only validates config/data, not actual model availability or target shape.

## Measured weight-edit contract

- Read docs/WEIGHT_EDITING.md before using the direct-edit API. Inspect metadata/prefix values without loading a model with `ouroboros weights inspect`.
- `run_weight_edit_trial` always restores its selected live parameter, including accepted trials, errors and cancellation. Only accepted candidates export a patch. Exit 0 can mean a measured rejection; inspect `passed`, `reason` and `model_restored`.
- `apply_weight_patch` is a separate explicit mutation; `reverse=True` requires the exact patched tensor. Hashes bind the target tensor, **not all model weights**, author identity or licensing rights.
- Same-process prediction/proposal/verifier callbacks are trusted. Trial rollback does not isolate other tensors/files; dense snapshots/sorts/hash copies can require large RAM. Use external quotas and isolation.
- Do not apply base edits inside PEFT SFT and assume adapter-only snapshots/checkpoints own them. Keep interventions separate unless their backend rollback/export contract is implemented and tested.
- `weights demo` is a **2×2 Torch classifier**. The training demo is a tiny NumPy classifier. Neither is an LLM/math benchmark or Astra reproduction.

## Same-model / MCP contract

- Read docs/SELF_EDITING.md and docs/MCP.md. Self-edit means THIS target model chooses a constrained menu label with its logits, not a full self-written weight program or guaranteed improvement. `require_attempt: true` is an explicit smoke-only forced-choice setting; API policy defaults to allowing abstention.
- Practice data only enter proposal prompts. Fixed validation drives strict original/latest gates; a final test runs once after selection. Accepted patch chains are used in memory then reversed in reverse order. Full state/source hashes check restoration; no base model files are overwritten.
- The checked-in pretrained report used a pinned Q4_1-derived 135M SmolLM2, not original full precision. Both nonzero trials plateaued/rejected. Do not call it an accuracy gain, semantic weight discovery, Astra reproduction or a benchmark.
- MCP defaults to local stdio read-only tools. Execution requires startup opt-in AND unchanged allowlisted configs. Do not add generic shell, download, delete, unrestricted file-read or generated-code execution tools. Path checks/offline child flags/timeouts are not an OS sandbox.
- Mark real SDK/protocol tests `mcp`, Torch tests `ml`; ordinary pytest must never implicitly download pretrained data. scripts/run_pretrained_smoke.py requires explicit --fetch or an existing receipt-verified model and a new workdir. CI uploads JSON evidence only.

## Research Lab contract (separate from the finite-menu loop)

- Read docs/RESEARCH_LAB.md and docs/PAPER_PARITY.md. `lab` is the frozen-editor closed-JSON research workflow: 19 actions / seven numeric operations, not general Python or full paper parity. Model mode really generates actions; scripted mode must always stay labelled as scripted. Never substitute scripted actions after invalid model output and call it autonomous.
- Target overlays are restored before each editor reply. Each session starts from the same original/damaged target; edits do not accumulate across sessions. Dense flat edits support ranks 1–32, matrix operations need rank 2; protection/shared-storage/rounded global budgets apply. No quantized/vision/vLLM/tensor-parallel claim.
- Only practice prompts/references and bounded practice measurements enter conversation. `submit` seals for one reserved check; only confirm/withdraw follows. Final data are opened after the incumbent is frozen. Raw held-out artifacts and operator damage specs exist on the trusted host, not in editor feedback; filesystem isolation is external.
- Research bundles bind complete before/after state (unlike legacy target-only patches). Data-only verification cannot reconstruct a model; explicit apply/reverse still needs an exact matching model and a dense-snapshot bound. No source overwrite or automatic bundle application in MCP/SFT.
- The initial unconstrained 135M research editor produced 0 valid / 2 invalid actions; preserve that historical negative result. See docs/EDITOR_EXPERIMENTS.md for the follow-up; syntax-constrained valid calls are not a beneficial edit. The separate 16-action scripted integration changed 1,152 entries / two tensors, reserved accuracy 0→0, rejected/restored. No pretrained capability gain or paper/WeightBench/standard-benchmark reproduction. Tiny accepted Torch fixtures are not LLM evidence.
- `lab diagnose` is practice-only; it must not run editor/edits/reserved/final inference. New scaffolds explicitly opt in to structured commands and separate editor sampling. Grammar filters known public ids/names, not edit correctness. Preserve strict grading/gates, honest `stalled` status and negative/incomplete pilot evidence. The source tokenizer EOS mismatch is aligned in memory; do not rewrite an existing model receipt.
- `lab import-data` requires explicit local reference-labelled data and source/license assertions; it does not verify them independently. `lab benchmark` is a plan-first offline command bridge to a separately installed trusted `lm-eval`, not an installed/reproduced benchmark or patched-model exporter.
- MCP has 11 read-only tools by default. Research execution needs opt-in plus `--lab-config`; approval binds config/data/script bytes. Smoke/research children share one lock. No generic shell, notebook, arbitrary mutation or hidden-label file tool.

## Discovery contract (opt-in research profile)

- Read docs/DISCOVERY_EXPERIMENTS.md. `init --discovery`, `lab tools --discovery` and `lab experiment` provide the broader five-role profile; the original lab remains 19 actions, discovery has 25. All eight hypothesis/gradient/compare/evaluate/rollback/analyze/record/stop tools are included. An implicit sweep/control cannot bypass fresh per-edit hypotheses/audits.
- Keep command validity, actual nonempty edit validity, strict useful accuracy gain, unseen generalization, post-intervention retention and improved future editing ability separate. Lower loss, passing tests, valid queries or host-authored controls are not autonomous capability gains.
- Practice→candidate-bound prospective hypothesis→actual numerical edit→real feedback→revision/rollback is adaptive; development seals before inference, reserved independently confirms, final/retention never reselect. Preserve strict significance/zero-domain-forgetting rules. The 44-submission family remains alpha 0.05/44 even when running subsets.
- `inspect_gradients` is explicit gradient assistance, never paper-parity gradient-free discovery. Host positive/random policies emit the SAME audited tools and must never substitute for failed model choices. Staged workflow constrains phases only, not model numerical decisions.
- Every attempted edit needs complete actual rounded before/after values/indices, tensor/base hashes, count/norm, validation, practice outcomes, gate results and replay/restoration. Failed/noop trials remain. A cache hit is not new inference or independent replication; record actual sequence/NLL counts separately.
- The five-role 240-example data include 60 fixed GSM8K selections with upstream MIT notice and row/byte hashes. Preserve the manifest, source/licensing attribution and exact grader. Do not call it full GSM8K or pretraining-decontaminated. MCP approvals bind all five roles.
- Freeze code/data/model/config before model loading. Do not change source under a running study: the driver will refuse the next arm. Never overwrite, silently resume or retarget a preregistered experiment. Preserve interrupted evidence and start a newly named study for a changed protocol.
- Useful-edit retention is null when no useful edit was accepted; four-cell baseline/diagnostic perturbations do not change that. A fixed small follow-up edit is not extended training. Within-context revision is not a new trained editing-policy checkpoint.

## Research foundation contract (separate from lab/discovery)

- Read docs/RESEARCH_FOUNDATION.md and papers/DEEPSEEK_APPLICABILITY.md before changing `ouroboros/research/`. This loop is bounded training research: the proposer may emit only versioned JSON hypotheses/recipes, and the controller/executor/evaluator remain distinct processes. A stronger researcher model improving a target is assisted improvement, not target self-improvement.
- Keep the evidence ladder explicit: infrastructure → beneficial host control → independent model proposals → useful autonomous edit → improved future selection. Only infrastructure-with-a-random-fixture is established. Never describe the constant-answer scaffold, a passing test or a valid JSON proposal as a capability or generalization result.
- Baselines are immutable, plans/designs/selections/artifacts are write-once, and SQLite is authoritative over any JSON or model narrative. Claims are stored `verified = 0`. Holdouts are consumed before inference and must never be reopened; a failed look still burns them.
- Never weaken `alpha = 0.05/family_size`, the per-seed strict-accuracy rule, zero-domain-forgetting, the separate robustness non-inferiority gate or the four-cell retention comparison. Practice accuracy chooses the recipe; validation and later roles never reselect. Loss, NLL, confidence and model prose never rescue a candidate.
- Do not add sequence packing, mid-step resume, MoE/MLA conversion, learned verifiers, FIM or multimodal paths by relabelling a nearby implementation. Supported families are `lora_sft`, `full_sft`, `data_reweight`, `distill_sft`; anything else needs its own design, gates and evidence.
- Data admission is not decontamination. The question-overlap screen covers questions only, not answers, semantics or pretraining data, and reports `semantic_decontamination: false`.
- Resource watchdogs (wall/RSS/output bytes/step and export budgets) and process separation are cooperative bounds, not an OS sandbox or hard quota. Do not claim hidden-data secrecy or adversarial isolation.
- Never resume, retarget or reuse the hidden sets of the frozen 135M/360M discovery studies. An interrupted foundation run needs a new plan and fresh confirmation.
- Verify with `ouroboros research run --config ... --check-only`, protocol tests, and `ouroboros research verify-run --report ...`; keep `scripts/run_foundation_fixture.py` labelled synthetic.

## Map

- ouroboros/self_edit.py, model_files.py, project.py, resources/: same-model controller, explicit model provisioning and installed-package quickstart fixtures.
- ouroboros/research/: bounded training-research foundation (contracts, controller, fixed workers, audited SFT, independent gates, SQLite memory, DeepSeek applicability matrix).
- ouroboros/lab/: separate frozen-editor research programs, instrumentation, controls, sealed validation/sessions, full-state-bound bundles and explicit dataset/benchmark boundaries.
- ouroboros/mcp_server.py: optional official-SDK stdio server, read-only default and allowlisted offline child execution.
- ouroboros/backends/: NumPy reference classifier, shared frozen LocalCausalLM loader and optional PEFT SFT backend.
- ouroboros/pipeline/self_loop.py: measured control flow, owned-state rollback, stage evidence and once-only post-selection final test.
- ouroboros/data.py: schema, provenance, replay and split boundaries.
- ouroboros/tensor_inspection.py: bounded read-only safetensors metadata/numeric prefixes without Torch.
- ouroboros/core/weight_trials.py: measured trials and data-only sparse patch verification/application/reversal.
- Other ouroboros/core/ modules: experimental arithmetic/architecture/proposal/evaluation/storage primitives.
- ouroboros/integrations/: explicit installed upstream calls, never fake fallbacks.
- external/sources.lock.json: seven pinned submodules and five preserved snapshots, with integration levels/licenses.
- papers/research.json: curated source-linked metadata; regenerate PAPERS_INDEX.md with scripts/update_paper_index.py.
- artifacts/manifest.json: starts empty; absent historical payloads are not recoverable from LFS patterns.

Keep new upstream experiments in separate environments, preserve licenses and state what was and was not run. Unit tests and fixtures do not establish pretrained-model capability gains. Use the release tag or active branch for installation while the current pull request is unmerged; do not claim the old default branch already contains this implementation.
