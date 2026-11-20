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

## Same-model experiments and MCP

The [self-edit controller](docs/SELF_EDITING.md) offers a host-defined, bounded numeric menu to the same target model; it does not execute model-generated programs. Validation/final-test questions and labels are not put into the proposal prompt. Practice data, model outputs and earlier decision summaries remain untrusted content. A constrained label preference is not proof of semantic weight understanding.

Accepted patches are applied temporarily in order and reversed in reverse order in `finally`, including final-test errors/cancellation. Full state-dict hashes check that the owned chain restored; they **detect**, not undo, arbitrary trusted callback writes elsewhere. Model/tokenizer source files are rehashed after a successful configured run. Hard process termination can prevent report finalization; no source model is saved over. Preflight byte/element limits are helpful, not a hardened malicious-config memory sandbox.

The optional [MCP server](docs/MCP.md) is **local stdio and read-only by default**. Experiment tools are absent unless the operator supplies both `--allow-experiments` and an explicit config allowlist. Config hashes must remain unchanged; all configured paths must be inside the workspace, without hidden/traversal/symlink components. These checks do not defend against concurrent filesystem replacement or authenticate every input byte. Model/data/library inputs are trusted. There are no shell/download/delete/arbitrary-patch tools.

Enabled runs use a fixed-argument offline-Hub child with a wall timeout and serialized execution. A subprocess inherits environment/filesystem privileges; HF offline flags are not an OS network firewall, and a timeout is not a RAM/VRAM or security sandbox. Use an isolated, least-privileged environment without credentials, a tiny workspace and a separate trusted grader for adversarial work. Reports may contain private practice examples/predictions; tool annotations and JSON schemas do not automatically redact them or prevent client-side prompt injection.

Explicit model provisioning refuses existing destinations and executable/pickle inputs, pins source revisions and records file checksums. The pinned GitHub route reads only model data/license from a clone, not its plugin code. Q4_1-to-float32 conversion does not recover original precision. License/source metadata and receipts are assertions plus integrity checks, not signatures or legal/security approval.

## Generated source

AST/syntax validation does **not** make Python safe. `NumericObjective` is a small arithmetic interpreter without attribute access, calls, imports, comprehensions or `eval`/`exec`; it is intentionally less powerful than a general loss-function language. Source proposals are exported as patches with `approved: false` and require review.

For experiments requiring execution, use an OS-isolated disposable container or microVM with **no credentials, no network, no host Docker socket, no writable host mounts**, a non-root user, a read-only filesystem, resource/time limits and a narrowly mounted candidate directory. Keep the trusted grader outside that environment. Example container policy is in [sandbox/README.md](sandbox/README.md); it has not been exercised in this sandbox, where Docker is unavailable. A subprocess or Python `-I` alone does not provide these protections.

## Filesystem and operational limits

Each run exclusively owns its UUID output/checkpoint directories. Do not let another process mutate them or input patch/model files during save/verify/apply/prune; filename/symlink rechecks do not defend against a hostile concurrent filesystem writer. Step/time budgets are cooperative, not a defense against a hung callback, GPU kernel or malicious program. Final-test inference may outlast the selection time budget.

Dense snapshots, sorting, norm computations and full-target hashing scale with matrix size, even when the serialized patch is tiny. Architecture copies can exceed RAM/VRAM. Use external process/container quotas for untrusted or expensive work. Header/sample limits on inspection are not a general hard resource sandbox.

External EasyEdit/mergekit/TRL/DGM/WeightBench tools have their own dependencies, downloads, permissions and security models. Follow pinned documentation in isolated environments. Inclusion in the source inventory is not a security endorsement or a promise of compatibility.
