# Security and experiment isolation

Ouroboros is an experimental training controller, **not** a security sandbox. Report vulnerabilities privately through the repository owner's available security contact or GitHub's private vulnerability reporting when enabled. Do not post API keys, private data or exploit-bearing model files in public issues.

## Defaults

- No model or dataset downloads in training/evaluation/check-only/MCP commands. `models fetch`, `models fetch-smoke` and `repos sync` are explicit network operations.
- Local safetensors model loading with `trust_remote_code=False`; NumPy loads use `allow_pickle=False`.
- Bounded numeric inspection and data-only patch verification do not load Torch, model code or pickle.
- No source from `external/` executes merely because it is cloned, inspected or listed.
- No automatic execution/application of generated Python patches, Git commits, merges or deployments.
- Trials always restore their selected live parameter; accepted patches still require explicit application.
- Pruning is restricted to owned, correctly named, hash-verified checkpoints; symlinked/unrecognized directories are not deleted. CLI deletion requires `--apply`.

Examples contain no API credentials. Reports can contain predictions, local paths, license/provenance assertions and other metadata; models can memorize private training data. Review reports/checkpoints before sharing rather than assuming they are automatically redacted.

## Trust assumptions and evaluation

Training data, independently supplied verifiers, model architecture/tokenizer implementations from installed libraries, and application callbacks must be trusted. A `verified` boolean is provenance, not a cryptographic or semantic proof. Hashes detect content mismatch but do not authenticate an author who can change both payload and manifest. Use signatures and separate access control if needed.

Do not give a proposer write access to validation/final-test labels, the controller, its promotion rules, checkpoint ownership markers or audit records. Do not let a self-judge approve arbitrary targets without task-specific checks. An optional `data.test` split is blocked from curricula and evaluated once after selection, but same-process callbacks can bypass API argument boundaries if allowed to read files. Normalized prompt checks do not detect semantic leakage. Keep final data genuinely independent and measure safety/non-target regressions; a tiny parameter edit can have broad consequences.

## Direct weight patches

Read [the weight-edit contract](docs/WEIGHT_EDITING.md) before using it. Trial isolation covers **only the selected tensor**, module training flags and Torch RNG. Trusted prediction callbacks must be deterministic and read-only; writes to other weights, buffers, files, Python/NumPy RNG or external services are not snapshotted. Do not run trials/apply patches during an in-flight optimizer/autograd step.

Patches are data-only safetensors plus JSON, with bounded file sizes, exact two-file inventories, index/dtype/finite checks and explicit before/after values. `verify-patch` checks their format/integrity, not their scientific validity. `apply_weight_patch` additionally checks full expected input/output **target-tensor** hashes, selected values and the norm policy before mutation. Those checks do not bind all other model weights, authenticate the author, validate asserted licenses or certify behavior in a different full model. Weight tying can give one edited tensor multiple roles.

Reversal needs the exact expected patched tensor; it is not a substitute for retaining an independent known-good model. Only ordinary little-endian F16/BF16/F32/F64 matrices are supported; quantized models are refused. PEFT adapter-only snapshots/exports do **not** preserve/undo arbitrary base-weight edits. Keep direct trials separate from SFT unless the corresponding backend state/export contract is implemented and tested.

## Research Lab numerical programs and bundles

The [Research Lab](docs/RESEARCH_LAB.md) uses model-generated **data-only JSON**, not general Python. A closed 18-action language can request only bounded numeric reads, supported tensor arithmetic, practice comparisons/controls and an explicit submit/confirm/withdraw protocol. Finite same-shape dense rank 1–32 edits have global changed-entry/fraction/byte and per-tensor norm gates; embeddings/output/final norm, distinct shared-storage parameters and quantized edits are refused for supported layouts. Architectural name checks are not a universal model-policy proof. No arbitrary paths, imports, file replacement, optimizer, generated shell or `exec` tool is exposed.

Only clean resident weights generate editor replies. Every requested target overlay is restored in `finally`; independent sessions reset to the same starting target. In repair mode the clean editor is **not** exposed through read tools: `original` means the damaged target, and damage location/specification stays operator-private. Optional attention extraction can allocate large intermediate arrays; bounds are cooperative allocation checks, not a hardened model-code memory sandbox.

After `submit`, the candidate is sealed for one reserved check; only aggregate feedback and confirm/withdraw follow. Final testing starts after the winner is frozen. Raw reserved/final measurements and the private damage specification **are written to trusted host artifacts**, never model observations; API argument separation does not make files inaccessible to hostile code in the same process. Do not share the whole output tree or expose its filesystem to the proposer. Grader libraries, callbacks, data and model implementations remain trusted. A hashed evaluation plan is not external preregistration or protection against repeated-validation overfitting.

Research bundles are a **different format** from legacy single-matrix patches: exact sparse before/after bytes, bounded two-file inventory and complete before/after **model-state** hashes. All named targets are preflighted before all-or-rollback application. Application/reversal requires the exact matching full state and an explicit dense-snapshot bound. Data-only `verify-bundle` cannot reconstruct a model to certify those state hashes or behavioral safety. The resident/source model is not overwritten, including after accepted sessions; explicit Python application is a separate trusted act, not an MCP tool. Keep an independent known-good base/license.

The local data importer trusts caller-supplied reference labels and license assertions; it never creates independently verified self-labels. The optional external benchmark driver is **plan-only by default**, never installs code/data and needs `--execute` to run the separately installed `lm-eval` executable. Fixed argument arrays, task-name checks, offline flags, new outputs and process-group timeout cleanup do not make upstream task/library code safe. PATH/executable provenance and any pre-provisioned benchmark data are operator responsibilities. Run upstream tools without secrets in isolated workers; offline environment flags are not a firewall.

## Same-model experiments and MCP

The legacy [self-edit controller](docs/SELF_EDITING.md) offers a host-defined, bounded numeric menu to the same target model; it does not execute model-generated programs. Research Lab supports the narrower data-only programs described above, never arbitrary Python. Validation/final-test questions and labels are not put into the proposal prompt. Practice data, model outputs and earlier decision summaries remain untrusted content. A constrained label preference is not proof of semantic weight understanding.

Accepted patches are applied temporarily in order and reversed in reverse order in `finally`, including final-test errors/cancellation. Full state-dict hashes check that the owned chain restored; they **detect**, not undo, arbitrary trusted callback writes elsewhere. Model/tokenizer source files are rehashed after a successful configured run. Hard process termination can prevent report finalization; no source model is saved over. Preflight byte/element limits are helpful, not a hardened malicious-config memory sandbox.

The optional [MCP server](docs/MCP.md) is **local stdio and read-only by default**. Experiment tools are absent unless the operator supplies both `--allow-experiments` and an explicit config allowlist. Config hashes must remain unchanged; research approvals additionally bind dataset/script bytes. All configured paths must be inside the workspace, without hidden/traversal/symlink components. These checks do not bind all model/library bytes, defend against concurrent filesystem replacement or authenticate inputs. Model/data/library inputs are trusted. There are no shell/download/delete/arbitrary-patch tools.

Enabled runs use a fixed-argument offline-Hub child with a wall timeout and serialized execution. A subprocess inherits environment/filesystem privileges; HF offline flags are not an OS network firewall, and a timeout is not a RAM/VRAM or security sandbox. Use an isolated, least-privileged environment without credentials, a tiny workspace and a separate trusted grader for adversarial work. Reports may contain private practice examples/predictions; tool annotations and JSON schemas do not automatically redact them or prevent client-side prompt injection.

Explicit model provisioning refuses existing destinations and executable/pickle inputs, pins source revisions and records file checksums. The pinned GitHub route reads only model data/license from a clone, not its plugin code. Q4_1-to-float32 conversion does not recover original precision. License/source metadata and receipts are assertions plus integrity checks, not signatures or legal/security approval.

## Generated source

AST/syntax validation does **not** make Python safe. `NumericObjective` is a small arithmetic interpreter without attribute access, calls, imports, comprehensions or `eval`/`exec`; it is intentionally less powerful than a general loss-function language. Source proposals are exported as patches with `approved: false` and require review.

For experiments requiring execution, use an OS-isolated disposable container or microVM with **no credentials, no network, no host Docker socket, no writable host mounts**, a non-root user, a read-only filesystem, resource/time limits and a narrowly mounted candidate directory. Keep the trusted grader outside that environment. Example container policy is in [sandbox/README.md](sandbox/README.md); it has not been exercised in this sandbox, where Docker is unavailable. A subprocess or Python `-I` alone does not provide these protections.

## Filesystem and operational limits

Each run exclusively owns its UUID output/checkpoint directories. Do not let another process mutate them or input patch/model files during save/verify/apply/prune; filename/symlink rechecks do not defend against a hostile concurrent filesystem writer. Step/time budgets are cooperative, not a defense against a hung callback, GPU kernel or malicious program. Final-test inference may outlast the selection time budget.

Dense snapshots, sorting, norm computations and full-target hashing scale with matrix size, even when the serialized patch is tiny. Architecture copies can exceed RAM/VRAM. Use external process/container quotas for untrusted or expensive work. Header/sample limits on inspection are not a general hard resource sandbox.

External EasyEdit/mergekit/TRL/DGM/WeightBench tools have their own dependencies, downloads, permissions and security models. Follow pinned documentation in isolated environments. Inclusion in the source inventory is not a security endorsement or a promise of compatibility.
