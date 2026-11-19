# Security and experiment isolation

Ouroboros is an experimental training controller, **not** a security sandbox. Report vulnerabilities privately through the repository owner's available security contact or GitHub's private vulnerability reporting when enabled. Do not post API keys, private data or exploit-bearing model files in public issues.

## Defaults

- No model or dataset downloads in core training/evaluation commands.
- Local safetensors model loading with `trust_remote_code=False`; NumPy loads use `allow_pickle=False`.
- No source from `external/` executes merely because it is cloned, inspected or listed.
- No automatic execution/application of generated Python patches, Git commits, merges or deployments.
- No secrets in example configuration or experiment logs.
- Pruning is restricted to owned, correctly named, hash-verified checkpoints; symlinked/unrecognized directories are not deleted. CLI deletion requires `--apply`.

## Trust assumptions

Training data, independently supplied verifiers, model architecture/tokenizer implementations from installed libraries, and application callbacks must be trusted. A `verified` boolean is provenance, not a cryptographic or semantic proof. Hashes detect accidental tampering but do not authenticate an author who can change both tensor and manifest. Use signatures and separate access control if needed.

Do not give a proposer write access to evaluation labels, the controller, its promotion rules, checkpoint ownership markers or audit records. Do not let a self-judge approve arbitrary training targets without task-specific checks. Keep a separate never-consulted final test set and measure safety/non-target regressions. A tiny parameter edit can have broad behavioral consequences.

## Generated source

AST/syntax validation does **not** make Python safe. `NumericObjective` is a small arithmetic interpreter without attribute access, calls, imports, comprehensions or `eval`/`exec`; it is intentionally less powerful than a general loss-function language. Source proposals are exported as patches with `approved: false` and require review.

For experiments requiring execution, use an OS-isolated disposable container or microVM with **no credentials, no network, no host Docker socket, no writable host mounts**, a non-root user, a read-only filesystem, resource/time limits and a narrowly mounted candidate directory. Keep the trusted grader outside that environment. Example container policy is in [sandbox/README.md](sandbox/README.md); it has not been exercised in this sandbox, where Docker is unavailable. A subprocess or Python `-I` alone does not provide these protections.

## Filesystem and operational limits

Each controller run exclusively owns its UUID checkpoint directory. Do not let another process mutate it during save/verify/prune; filename/symlink rechecks do not defend against a hostile concurrent owner of the same filesystem. Time/step budgets are cooperative, not a defense against a hung callback, GPU kernel or malicious program. Dense copies for architecture experiments and full-size tensor operations can exceed RAM/VRAM. Run under external process/container quotas for untrusted or expensive work.

External EasyEdit/mergekit/TRL/DGM/WeightBench tools have their own dependencies, downloads, permissions and security models. Follow their pinned documentation in isolated environments. Never assume that inclusion in the source inventory is a security endorsement or a promise of compatibility.
