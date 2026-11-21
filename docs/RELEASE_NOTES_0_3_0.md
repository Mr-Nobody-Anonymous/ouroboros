# Ouroboros 0.3.0 — inspect, measure and reverse weight edits (pre-release)

This extends the 0.2.0 audit/training foundation. It is an experimental toolkit release, not an Astra reproduction, a claim of general autonomous intelligence growth, or a pretrained capability benchmark win.

## Added

- Read-only safetensors metadata and bounded numeric samples without loading Torch or a model. Supports common floating/integer/bool formats, including BF16 prefixes; rejects oversized headers, invalid layouts and pickle files.
- Explicit local weight-edit trials with measured baseline/candidate predictions, per-domain counts/retention gates, exact entry limits, actual rounded norm limits and sparse-patch byte budgets.
- Guaranteed restoration of the selected live parameter after acceptance, rejection, exceptions, cancellation and export failure. Accepted trials export data-only patches; applying them is a separate explicit operation.
- Portable little-endian F16/BF16/F32/F64 patch data with full before/after target-tensor fingerprints, old/new values, integrity checks, exact application/reversal and refusal of mismatched targets or budgets.
- `weights inspect`, `weights demo`, `weights trial`, `weights verify-patch` and `weights evaluate-patch` commands; JSON config and runnable mechanical fixture.
- Shared offline local causal-LM inference/loading between direct-edit experiments and PEFT SFT; no remote code, Hub downloads or implicit LoRA insertion into direct trials.
- Optional `data.test` split, disjoint from train/replay/validation, excluded from proposals and promotion. It is evaluated once after selection; failures are recorded and produce nonzero CLI exit without falsifying the selected checkpoint.
- Durable curriculum-stage failures and adapter/RNG rollback when a proposer, verifier or conflicting generated label fails.

## Correctness and documentation

Preserve unedited F64 values and signed-zero bytes instead of round-tripping through F32 or adding zeros to every entry; validate complete prediction vectors before scoring so truncated `zip` output cannot inflate metrics. The weight-edit guide, architecture, training, agent instructions and research map distinguish these APIs from automatic causal diagnosis or learned outer-loop RL.

Local Python 3.11 verification returned **195 passed, 3 expected PEFT warnings**; a fresh no-Torch/Transformers/PEFT environment returned **168 passed, 27 ML tests deselected**. An extracted sdist passed **167 tests** with one appropriate Git-provenance skip; a non-editable wheel ran the demo/reload/final-test smoke successfully. The [testing guide](TESTING.md) records scope and limits. Tests include four tensor dtypes, byte-exact patch undo, bad fingerprints/indices/norms, callback/cancellation/disk failures, final-test isolation and a real offline random GPT-2 trial. No pretrained model or benchmark corpus was downloaded. Check remote CI separately rather than inferring it from local tests.

## Distribution and limits

The core archive includes original source, small examples, documentation, tests and source references, **not upstream trees, models, caches or a recovered 600 MB payload**. The existing twelve-source inventory and 33-paper catalogue are retained; more cloned folders would not substitute for measured integration.

Quantized/direct GPU editing, full MATH-500/GSM8K/IFEval reproduction, learned SEAL/EvoTIDE/DPO/GRPO and safe arbitrary generated-code execution remain unvalidated or unimplemented. Patch hashes bind a target tensor, not every weight in another supplied model, author identity or redistribution rights. Read [weight-edit limits](WEIGHT_EDITING.md) and [security](../SECURITY.md) before use.

The historical v0.1.0 and the v0.2.0 pre-release remain unchanged. The release-only GitHub workflow can attach the new core archive/checksum from this tag; only the actual asset list and successful upload job establish publication.
