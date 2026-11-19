# Self-evolving architecture: implemented control flow and research boundaries

As of 2026-11-19, version 0.2.0. Read the [capability table](../README.md#capability-status) before interpreting “self-evolving.” This repository provides a bounded experimental controller, not a guarantee that a model can improve itself indefinitely.

## 1. Persistent state and trust boundaries

```text
local, licensed train data + independently verified replay anchors
                         |
                         v
             validate / deduplicate / fingerprint
                         |
optional proposer -------+------> independent proposal verifier
(no validation labels)           reject leaked or unverified data
                         |
                         v
               measure generation-0 baseline
                         |
                         v
     snapshot adapter + RNG -> train candidate under budgets
                         |                 |
                         |           failure/NaN/export error
                         v                 |
       predict on fixed validation         |
                         |                 |
             exact-match domain gate       |
                  /               \        |
               accept           reject <---+
                 |                |
       atomic tensor export    restore actual adapter + RNG
                 |                |
                 +-------+--------+
                         v
                 append run evidence
                         |
       Pareto retention of owned, hash-verified checkpoints
                         |
                next bounded generation
```

`DatasetBundle` rejects normalized prompt overlap between training/replay and validation. This is **exact normalized overlap checking**, not semantic decontamination. `Sample` retains source, domain, verification and anchor fields. Data fingerprints cover ordered records and provenance. An externally supplied proposer sees training examples and a selected weak-domain name, not validation labels; it still must be trusted not to read files directly.

## 2. Actual optimization paths

### NumPy reference backend

A frozen zero base plus `alpha/r * B @ A` maps training-derived bag-of-token features to training-derived labels. Analytical cross-entropy gradients update A/B. A dedicated NumPy RNG controls initialization and shuffling. It is deliberately a classifier, not a token-generating language model. Its tiny fixed demo is evidence of functioning plumbing only.

### Transformers + PEFT backend

A local safetensors causal LM is wrapped with installed PEFT LoRA. The backend masks prompt and padding labels with `-100`, retains completion labels/EOS, rejects completions that cannot fit, accumulates gradients weighted by completion-token count, clips gradient norms, and keeps base parameters frozen. Checkpoint export saves adapters and tokenizer files. Optimizers reset each generation; this is **not mid-step optimizer resume**. Independent runs use explicit seeds; cross-device bitwise equivalence is not promised.

Base weights are not copied into each checkpoint. Keep the original model directory and record its license/revision/checksum separately. Saved backend metadata records its local path; `evaluate --model-path` can override a relocated base path. GPU QLoRA is an optional path, not a tested hardware claim.

## 3. Promotion and rollback

Domain scores are normalized exact-match fractions. A candidate needs positive macro-domain gain meeting `improvement_threshold`. For each domain it must remain within `forgetting_threshold` of **both** generation 0 and the latest accepted generation. Missing domains, NaN/Inf, missing loss, zero optimizer steps, and checkpoint-export failure cannot promote a candidate. Equality is not progress. The weakest domain is used for diagnosis; all supplied training domains remain in the curriculum.

Rollback restores real adapter arrays/tensors and backend RNG state, not just displayed scores. Baseline/current checkpoints are protected from automatic deletion. Each run gets its own UUID directories so a rerun cannot overwrite another experiment.

**Statistical limitation:** the built-in gate is a deterministic engineering gate, not a significance test. Reusing validation over generations adapts to it. For scientific claims, reserve a separate never-consulted final test set, pre-register comparisons, run multiple seeds, report uncertainty/sample counts and side effects, and compare equal compute. Accuracy on public examples is not proof of general intelligence or robust unlearning.

## 4. Optional proposal and intervention surfaces

- **Curriculum:** inject `proposal_provider(domain, training_samples)` and `proposal_verifier(sample)` into `RecursiveTuningLoop`. At most 32 proposals are consumed per generation; the curator excludes leaked/overlong records and conflicting labels. A boolean verifier supplied by a caller must really check the task.
- **Preferences:** `SelfRewardingEngine` uses explicit judge/generator callbacks and abstains on low margins. Preference export is not DPO training; TRL and its research sources are separately documented.
- **Adversary:** `AdversarialCritic.challenge_round` scores canonical answers and returns verified failure examples. The built-in questions are fixtures, not learned adversarial agents. Do not train on the same challenges you intend to call held-out tests.
- **Test-time compute:** Best-of-N and majority-vote helpers require actual generation/verifier callbacks. High judge scores alone never turn an incorrect final answer into a supervised target.
- **Direct weights:** `WeightSpaceProgrammer` measures contrastive activation directions or exposes a plainly labeled weight SVD direction. Its sparse edit has an exact maximum entry budget, including ties and zero. PyTorch edits return rollback tensors. Singular values alone do not identify a causal repetition circuit.
- **Architecture:** `build_layer_stacked_candidate` explicitly copies a nonquantized Llama/Qwen-style decoder, updates its layer count/cache indices and leaves the original untouched. This is not automatic validated architecture search; memory, tokenizer, positional encoding and checkpoint compatibility remain the caller's responsibility.
- **Code/loss:** `CodeMutator` exports syntax-checked review patches with base/proposed hashes. No Python is executed or merged. `NumericObjective` interprets a small bounded arithmetic grammar without `eval`/`exec`; it is a scalar experiment utility, not an autodiff loss compiler wired into SFT.

These interventions are **not all applied sequentially** by default. Blindly combining a merge, architecture mutation and ablation can destroy a model and make causal attribution impossible. Evaluate one well-defined candidate intervention at a time.

## 5. Storage and provenance

A checkpoint is written to a temporary sibling, populated with actual tensors, hashed, verified, then renamed. The ownership marker records run ID, generation, parent, scores, backend metadata and per-file SHA-256. Hashes detect accidental changes, not malicious authors who can rewrite the manifest.

Pareto pruning compares macro accuracy (maximize) against actual bytes (minimize), caps the frontier, and always retains protected points. Only correctly named, verified, same-run directories are eligible; foreign, unknown, corrupt or symlinked paths are skipped. The nominal cap can be exceeded by protected points. The CLI previews by default. The controller prunes only its private checkpoint directory after persisting a decision. Cooperative time checks cannot interrupt an in-flight GPU kernel or a hung external callback.

## 6. Research-to-implementation map

| Research direction | Evidence and limits | Local surface |
| --- | --- | --- |
| LoRA / QLoRA | [Original papers](../papers/LoRA_and_QLoRA.md); resource claims depend on setup | Real PEFT SFT; GPU quantization untested |
| SEAL | [2506.10943](https://arxiv.org/abs/2506.10943): learned self-edit outer RL | Verified proposal interface + local SFT, not SEAL RL |
| Weight-space programming | [MIT post](../papers/Sparks_of_Weight_Space_Programming.md): specific repetition repair | Bounded reversible primitives; not reproduced |
| Self-rewriting | [2611.21885v2](https://arxiv.org/abs/2611.21885v2): larger model gains, smaller models fail | Measured candidate/abstention/rollback design |
| Harness/weight co-evolution | [Evo2](https://arxiv.org/abs/2610.13025), [SIA](https://arxiv.org/abs/2605.27276) | Separate reviewable patch and adapter experiments, not joint RL |
| Evaluation and continual learning | [WeightBench](https://arxiv.org/abs/2611.18553), [continual-agent study](https://arxiv.org/abs/2610.25858) | Independent source checkouts, original/latest gates |
| Recursive synthetic-data drift | [Nature article](https://www.nature.com/articles/s41586-024-07566-y) | Verified replay mitigation, not a universal ratio or theorem |

See the [recent review](../papers/RECENT_RESEARCH_2026.md) for primary-source dates, review scope and counterevidence.
