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

`python scripts/install.py --profile core --dev` creates the local environment with optional MCP; `--profile llm` adds same-model inference/GGUF smoke provisioning, `--profile research` adds the explicit Math-Verify grader, and `--profile train` adds PEFT. `ouroboros init NEW_DIRECTORY` packages its own tiny fixtures. Run `self-edit --config CONFIG --check-only` before a real run. Model downloads are explicit `models fetch` (Hub) / `models fetch-smoke` (pinned GitHub derivative), not side effects of inference.

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

- Read docs/RESEARCH_LAB.md and docs/PAPER_PARITY.md. `lab` is the frozen-editor closed-JSON research workflow: 18 actions / seven numeric operations, not general Python or full paper parity. Model mode really generates actions; scripted mode must always stay labelled as scripted. Never substitute scripted actions after invalid model output and call it autonomous.
- Target overlays are restored before each editor reply. Each session starts from the same original/damaged target; edits do not accumulate across sessions. Dense flat edits support ranks 1–32, matrix operations need rank 2; protection/shared-storage/rounded global budgets apply. No quantized/vision/vLLM/tensor-parallel claim.
- Only practice prompts/references and bounded practice measurements enter conversation. `submit` seals for one reserved check; only confirm/withdraw follows. Final data are opened after the incumbent is frozen. Raw held-out artifacts and operator damage specs exist on the trusted host, not in editor feedback; filesystem isolation is external.
- Research bundles bind complete before/after state (unlike legacy target-only patches). Data-only verification cannot reconstruct a model; explicit apply/reverse still needs an exact matching model and a dense-snapshot bound. No source overwrite or automatic bundle application in MCP/SFT.
- The 135M research editor produced 0 valid / 2 invalid actions. The separate 16-action scripted integration changed 1,152 entries / two tensors, reserved accuracy 0→0, rejected/restored. No pretrained capability gain or paper/WeightBench/standard-benchmark reproduction. Tiny accepted Torch fixtures are not LLM evidence.
- `lab import-data` requires explicit local reference-labelled data and source/license assertions; it does not verify them independently. `lab benchmark` is a plan-first offline command bridge to a separately installed trusted `lm-eval`, not an installed/reproduced benchmark or patched-model exporter.
- MCP has 11 read-only tools by default. Research execution needs opt-in plus `--lab-config`; approval binds config/data/script bytes. Smoke/research children share one lock. No generic shell, notebook, arbitrary mutation or hidden-label file tool.

## Map

- ouroboros/self_edit.py, model_files.py, project.py, resources/: same-model controller, explicit model provisioning and installed-package quickstart fixtures.
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
