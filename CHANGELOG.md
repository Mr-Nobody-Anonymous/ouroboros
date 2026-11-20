# Changelog

## 0.4.0 — same-model experiments and local MCP (development)

### Added

- A real small causal LM uses its own next-token logits to choose a bounded host-defined edit to its own MLP weights. Optional abstention, original/latest retention, strict positive-gain gates and isolated final-test inference remain host-controlled.
- Accepted patches can form an in-memory chain; reverse-order cleanup, full state-dict restoration fingerprints and source-file rehashing leave the original base model unchanged.
- Explicit model provisioning with resolved revisions, byte gates and SHA-256 receipts. A pinned/hash-checked GitHub Q4_1 SmolLM2 source is converted to float32 safetensors without executing its plugin; the original precision is not recovered. Separate Hub download path has mocked boundary tests.
- Packaged `init` fixtures, three-profile venv installer, `llm`/`mcp`/`smoke` extras, improved doctor, JSON CLI and explicit pretrained end-to-end smoke script.
- Official-SDK **stdio MCP**, read-only by default. Optional experiments need startup opt-in plus unchanged allowlisted configs, workspace path checks, serialized fixed-argument offline-Hub children and wall-time/cancellation cleanup. No generic shell/download/delete/apply tool.
- Local pretrained evidence, offline/random-LM and real MCP protocol regressions, plus dedicated GitHub MCP/pretrained smoke jobs with small JSON artifacts only.
- One compact Ouroboros emblem, a new quickstart-focused README and mechanism/permissions/security/testing guides.

### Fixed

- Preserve original-generation retention floors across successive direct trials rather than permitting threshold-sized incremental drift.
- Share reviewed chat-template rendering between inference, proposals and SFT; suppress unsupported tokenizer `token_type_ids` for portable causal-LM generation.
- Handle scalar buffers in full-state hashing, malformed model IDs, shard traversal, changed model receipts, failed chain reversal and failed/interrupted final-test states.
- Drain child pipes during timeout/cancellation cleanup to avoid waiting on a full stdout buffer.

### Observed result and limits

The actual 134,515,008-parameter pretrained Q4_1-derived SmolLM2 selected two nonzero edits (884 entries each). Both plateaued on the four-example strict exact-match fixture and were rejected/restored; no patch or capability gain is claimed. The metric penalizes verbose/truncated answers. This is constrained menu choice, not free-form self-written weight programs, semantic circuit discovery, Astra replication, statistical evidence or an OS sandbox. Historical published releases are unchanged. See [self-edit evidence](docs/SELF_EDITING.md) and [MCP permissions](docs/MCP.md).

## 0.3.0 — measured reversible weight experiments (2026-11-19, pre-release)

### Added

- Bounded read-only safetensors metadata/value inspection, including BF16/nonfinite handling, without Torch/model loading.
- Explicit single-matrix trials with paired baseline/candidate inference, measured improvement/retention gates and actual rounded entry/norm/payload-byte limits.
- Unconditional selected-parameter/flags/Torch-RNG restoration, accepted-only data-only sparse patch exports and exact target-bound application/reversal APIs.
- `weights inspect`, `demo`, `trial`, `verify-patch` and `evaluate-patch` CLI commands, JSON templates and a real 2×2 Torch mechanical fixture.
- Shared frozen offline `LocalCausalLM` loader/inference for both direct trials and PEFT SFT.
- Optional disjoint `data.test`, blocked from proposals/promotion and evaluated once after completed/budget-limited selection, with explicit failure/skipped/not-configured states.
- Weight-edit documentation and refreshed architecture/training/security/storage/agent/research implementation guidance.

### Fixed

- Validate complete prediction sequences before exact-match scoring; reject wrong cardinality/types rather than silently truncating `zip`.
- Preserve unedited F64 values/signed-zero bytes, including non-C-layout inputs; use at least F32 arithmetic for low-precision weights and measure the actual rounded result.
- Enclose curriculum callbacks/validation in the generation rollback boundary; persist terminal proposer/verifier/conflicting-label failures and last-stage evidence.
- Keep completed training metrics when candidate evaluation fails.
- Fail closed on malformed, extra-file, corrupt, mismatched, over-budget or rehashed inconsistent patch inputs before live application.

### Limits

These are mechanical/tooling improvements, not pretrained capability gains or an Astra replication. Direct trials own one tensor, not arbitrary callback side effects; patch hashes bind that target, not the whole model/author/license. Adapter-only checkpoints cannot capture arbitrary base-weight edits. Dense RAM costs, trusted callbacks, quantized/GPU limitations and repeated-validation overfitting remain explicit. See [release notes](docs/RELEASE_NOTES_0_3_0.md) and [testing](docs/TESTING.md).

## 0.2.0 — measured adaptation and reproducibility (2026-11-19)

### Added

- Installable minimal Python package, optional local ML extras, diagnostic/configuration CLI and CI.
- Real CPU NumPy LoRA-classifier optimization plus optional offline Transformers/PEFT causal-LM SFT.
- Validated provenance-bearing data, normalized leakage/dedup checks, independently verified proposal interface and correctly proportioned replay.
- Measured original/latest promotion gates, actual parameter/RNG rollback, persistent candidate evidence and bounded experiments.
- Atomic tensor checkpoints with SHA-256 inventories, lineage and conservative accuracy/actual-byte Pareto retention.
- TIES reference arithmetic, exact sparse budgets, reversible PyTorch candidate edits and explicit copied decoder-layer candidates.
- Reviewable AST-targeted code patches, bounded arithmetic objective interpreter, and optional container isolation policy.
- Five pinned research/evaluation sources in addition to repaired EasyEdit/mergekit submodules and five preserved snapshots.
- Thirty-three source-linked references, eleven recent primary-source reviews, machine-readable catalogue/BibTeX and corrected Markdown documentation.
- Honest empty artifact inventory, narrow LFS attributes, artifact registration/checksums and deterministic core source release builder.

### Fixed / removed misleading behavior

- Random score gains/gradients masquerading as LLM training, fake EasyEdit success, metadata-only checkpoints and cosmetic rollback.
- Missing `.gitmodules`, unbounded tied sparse masks, zero-ratio mistakes, missing shape/finite checks and destructive unknown-folder pruning.
- Keyword-based self-rewards, random verifier values and acceptance of wrong answers solely due to confidence.
- Incorrect MoE-CL arXiv attribution, conflated TIES/DARE citations, vague SIA reference, universal replay/no-forgetting claims and false release/LFS assumptions.

### Explicit limitations

No pretrained capability benchmark, Astra/Qwen reproduction, learned SEAL/EvoTIDE outer RL, default DPO/GRPO trainer, universal self-judge, automatic trusted-source merge, hard sandbox, or GPU QLoRA validation is claimed. See [audit](docs/AUDIT_AND_MIGRATION.md) and [test evidence](docs/TESTING.md).

## 0.1.0 — original demonstration

Historical source release. The starting loop simulated improvements and the published release had no uploaded assets when audited. Older README/release claims should not be interpreted as verified benchmark or artifact-distribution evidence.
