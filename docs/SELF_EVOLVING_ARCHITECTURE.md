# Self-evolving architecture: implemented control flow and research boundaries

As of 2026-11-19, version 0.5 development. Read the [capability table](../README.md#capability-status) before interpreting “self-evolving.” This repository provides a bounded experimental controller, not a guarantee that a model can improve itself indefinitely.

## 1. Persistent state and trust boundaries

```text
local, licensed train data + independently verified replay anchors
fixed validation split; optional disjoint final-test split
                         |
                         v
             validate / deduplicate / fingerprint
                         |
                         v
               measure generation-0 baseline
                    on VALIDATION only
                         |
                         v
             snapshot owned adapter + RNG
                         |
optional proposer -------+------> independent proposal verifier
(training records only)          block validation AND test overlaps
                         |
                         v
               train under cooperative budgets
                         |                 |
                         |        curriculum/train/eval/export error
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
                         |
                         v
          selection completed / budget exhausted
                         |
             final test ONCE, if configured
          (not a promotion or rollback decision)
```

`DatasetBundle(train, validation, anchors, test)` rejects normalized prompt overlap between train/replay and validation, and between final test and **all** selection splits. This is exact normalized overlap checking, not semantic decontamination. `Sample` retains source, domain, verification and anchor fields. Data fingerprints cover ordered records and provenance; an absent test split preserves the 0.2 fingerprint payload.

An externally supplied proposer sees training examples and a selected weak-domain name, not validation/test labels. Proposals overlapping either held-out split are blocked before verification. Callbacks still execute in the same trusted process and must not read labels or mutate other state directly; API argument separation is not OS isolation.

## 2. Actual optimization and inference paths

### NumPy reference backend

A frozen zero base plus `alpha/r * B @ A` maps training-derived bag-of-token features to training-derived labels. Analytical cross-entropy gradients update A/B. A dedicated NumPy RNG controls initialization and shuffling. It is deliberately a classifier, not a token-generating language model. Its tiny fixed demo is evidence of functioning plumbing only.

### Shared offline causal-LM loader and PEFT backend

`LocalCausalLM` loads a local safetensors model/tokenizer without remote Python or implicit downloads, freezes its weights, hashes standard base/tokenizer files and provides greedy inference. It does not insert PEFT adapters. `TransformersLoRABackend` builds on this loader with installed PEFT LoRA and the same explicit prompt template.

SFT masks prompt and padding labels with `-100`, retains completion labels/EOS, rejects completions that cannot fit, accumulates gradients weighted by completion-token count and clips gradient norms. Base parameters stay frozen. Checkpoint export saves adapters and tokenizer files. Optimizers reset each generation; this is **not mid-step optimizer resume**. Independent runs use explicit seeds; cross-device bitwise equivalence is not promised.

Base weights are not copied into each checkpoint. Keep the original model directory and record its license/revision separately. Saved metadata records local path and file hashes; `evaluate --model-path` supports relocation but refuses recorded content mismatches. GPU QLoRA is an optional training path, not a tested hardware claim. Direct weight trials refuse quantized models.

### Same-model constrained weight selection (0.4)

`self_edit.py` is a **separate controller**, not a silent insertion into PEFT SFT. It measures a frozen local-model baseline, constructs at most four explicit coordinate/strength ablations, and obtains a constrained next-token choice from that **same target model**. Practice examples/predictions and previous edit/decision summaries enter the prompt; held-out questions/labels do not. A label preference is not evidence of semantic weight understanding. `require_attempt` removes abstention only when deliberately requested (the smoke fixture does so).

Each trial always restores its one matrix and enforces entry/norm/byte budgets. An accepted data-only patch is then applied in memory; original/latest domain floors prevent cumulative retention drift. Final-test inference occurs once after selection, then the accepted chain is reversed in reverse order even on final-test failure/cancellation. Full state-dict fingerprints (including buffers) check restoration; successful configured runs rehash base/tokenizer files. These checks detect foreign callback changes but do not snapshot or undo arbitrary side effects. No full model overwrites the source.

`project.py`/packaged resources provide `init` from a wheel. `model_files.py` separates **explicit** pinned/size-bounded model provisioning from local-only inference. Its GitHub smoke route was actually cloned and adapted as data-only Q4_1-to-float32 conversion, not plugin execution or recovered full precision. The pretrained 135M experiment rejected two nonzero plateau edits; see [mechanism/evidence](SELF_EDITING.md).

`mcp_server.py` uses the optional official SDK over **stdio only**, read-only by default. It exposes bounded inspection/config/report tools; execution appears only with startup opt-in and unchanged allowlisted configs. A fixed offline-Hub child, serialized runs and a wall timeout are operational controls, not an OS sandbox. See [permissions/threat model](MCP.md).

### Frozen-editor Research Lab (0.5)

[`ouroboros/lab/`](../ouroboros/lab) is a third, distinct path. It does **not** mutate the finite-menu or PEFT controller into the paper's protocol. See [usage](RESEARCH_LAB.md) and the [source-linked parity audit](PAPER_PARITY.md).

```text
local clean safetensors → frozen resident editor → one JSON action per reply
                                  ↑                         |
                        restore owned overlays             v
practice only → diagnosis → target-copy reads / numeric programs / paired tests
                                  |                         |
                       damaged target optional       saved variants / controls
                                                            |
                                         seal → reserved check → confirm/withdraw
                                                            |
                                 independent base-reset sessions / host incumbent
                                                            |
                               freeze selection → once-only new-seed final test
                                                            |
                             exact bundle / full-state and source-integrity evidence
```

- `config.py` enforces a separate research profile, signed/finite settings, disjoint required splits and config/data/script content binding for MCP. `tools.py` defines **18 closed-language actions and seven numerical operators**, not a Python execution API.
- `backend.py` extends local inference with explicit sampling, presence/repetition penalties, thinking closure, EOS/cut records, per-problem seeds and RNG/mode preservation. Generation prompts cannot be silently truncated. Only serial single-device Transformers is implemented; no vLLM, tensor parallelism or vision loader.
- `state.py` owns temporary target overlays and private damage; `program.py` atomically builds multi-tensor rounded programs under global entry/fraction/byte and per-tensor norm limits. Generic flat edits/clipping work on dense rank 1–32 parameters; matrix-specific operations stay rank 2. Protected embeddings/output/final-norm storage, unsupported quantization and distinct shared-storage parameters fail closed. No whole-tensor rescaling.
- `instrumentation.py` measures actual hidden states, MLP inputs, token logprobs and opt-in attention, with selected modules, pooling/last/stride, allocation bounds and explicit read truncation. Measured contrast vectors and column ranking aid experiments; they do not establish semantic circuits.
- `session.py` meters tool requests/edits/turns/time, keeps notes and four alternatives, supports seeded original/edited/saved comparisons, random/no-op/layer controls and independent strength sweeps. Only practice references and bounded practice outputs enter conversation.
- `submit` irreversibly seals a session for one private reserved check; only `confirm` or `withdraw` follows. The host still rejects plateau, original/incumbent retention failures and optional significance failures. `runner.py` starts every session from the **same** target base, picks the best eligible incumbent without merging, fixes an evaluation plan before diagnosis and evaluates final data once after selection. No accepted edit is left on the resident editor.
- `metrics.py` includes explicit Math-Verify final-box grading and problem-clustered paired deltas/SE/sign-flips; repeated answers are averaged within problem. This is not automatic multiple-testing/sequential correction, externally timestamped preregistration or semantic decontamination. SFT/menu still use their simpler engineering gates.
- `patches.py` emits exact-byte sparse **multi-tensor bundles bound to the full before/after model state**, unlike the older target-only format. It verifies all targets before all-or-rollback application and supports exact explicit reversal or temporary use. Integrity is not author authentication, licensing approval or behavioral safety.
- `benchmarks.py` converts explicit local labelled datasets with receipts and plans a fixed offline external `lm-eval` invocation; execution needs opt-in and a separately installed trusted tool/data. No upstream benchmark result, patched-model export or paper sampling parity is implied.

CLI and MCP expose this same whole-run engine; MCP shares one execution lock for both legacy/research children. General Python execution is deliberately absent, not called a sandbox. Same-process callbacks/libraries/inputs remain trusted; full hashes can detect foreign side effects but cannot undo them. Cooperative limits do not bound in-flight kernels or memory. See [SECURITY.md](../SECURITY.md).

Actual local evidence: the pretrained 135M editor produced two invalid actions, **no useful program**. Separate scripted integration edited 1,152 entries across two tensors, measured 0→0/rejected, restored. Tests also cover a mechanically accepted tiny Torch classifier, explicitly **not** an LLM. Neither establishes a pretrained gain or reproduction of the 27B/math/vision studies.

## 3. Promotion, rollback and final tests

Domain scores are normalized exact-match fractions. `evaluate_samples` validates prediction cardinality and string types **before** scoring, then returns per-domain counts/scores, macro accuracy and predictions. A candidate needs positive macro-domain gain meeting `improvement_threshold`. For each domain it must remain within `forgetting_threshold` of **both** generation 0 and the latest accepted generation. Missing domains, NaN/Inf, missing loss, zero optimizer steps and checkpoint-export failure cannot promote a candidate. Equality is not progress. The weakest domain is used for diagnosis; all supplied training domains remain in the curriculum.

The snapshot/exception boundary encloses curriculum construction, training, evaluation and checkpoint export. A proposer/verifier error or conflicting generated target is a recorded terminal generation failure, not an uncaught silent partial run. Run history records the last stage reached. Rollback restores the backend's **owned** adapter arrays/tensors and RNG, not just displayed scores; it cannot undo arbitrary trusted callback side effects outside that state contract. Cancellation is persisted and reraised. Baseline/current checkpoints are protected from automatic deletion. Each run has UUID directories so reruns cannot overwrite another experiment.

With `data.test` configured, the selected live model is evaluated on it exactly once after selection completes or the generation budget is exhausted. Test scores do not promote/undo candidates or feed proposals. Training/curriculum failures skip it. A final-test error sets `final_test_failed` and a nonzero CLI exit without erasing the accepted checkpoint; interruption is persisted and reraised. No split is represented as passed when none was configured. Final inference may outlast the cooperative selection budget; external quotas are needed for a hard time limit.

**Statistical limitation:** the SFT/finite-menu gate is a deterministic engineering gate, not a significance test. Research Lab optionally adds a problem-clustered reserved significance gate, but that is not a correction for repeated session selection. Reusing validation over generations adapts to it. For scientific claims, use genuinely unseen final data, pre-register comparisons, run multiple seeds, report uncertainty/sample counts and side effects, and compare equal compute. Rerunning while adjusting to final-test results invalidates the intended separation. Public examples are not proof of general intelligence or robust unlearning.

## 4. Optional proposal and intervention surfaces

- **Curriculum:** inject `proposal_provider(domain, training_samples)` and `proposal_verifier(sample)` into `RecursiveTuningLoop`. At most 32 proposals are consumed per generation; the curator excludes leaked/overlong records and conflicting labels. A boolean verifier supplied by a caller must really check the task.
- **Preferences:** `SelfRewardingEngine` uses explicit judge/generator callbacks and abstains on low margins. Preference export is not DPO training; TRL and its research sources are separately documented.
- **Adversary:** `AdversarialCritic.challenge_round` scores canonical answers and returns verified failure examples. The built-in questions are fixtures, not learned adversarial agents. Do not train on challenges you intend to call held-out tests.
- **Test-time compute:** Best-of-N and majority-vote helpers require actual generation/verifier callbacks. High judge scores alone never turn an incorrect final answer into a supervised target.
- **Direct weights:** `WeightSpaceProgrammer` reduces caller-supplied contrastive activations or exposes a plainly labeled weight SVD direction. Neither automatically identifies a causal repetition circuit. `run_weight_edit_trial` tests a caller-specified direction/parameter with count/norm/byte and measured validation/retention gates, **always restores** the live target, and exports only accepted sparse patches. Explicit application/reversal is a separate policy decision, bound to complete before/after target-tensor hashes. See [weight editing](WEIGHT_EDITING.md).
- **Architecture:** `build_layer_stacked_candidate` explicitly copies a nonquantized Llama/Qwen-style decoder, updates its layer count/cache indices and leaves the original untouched. This is not automatic validated architecture search; memory, tokenizer, positional encoding and checkpoint compatibility remain the caller's responsibility.
- **Code/loss:** `CodeMutator` exports syntax-checked review patches with base/proposed hashes. No Python is executed or merged. `NumericObjective` interprets a small bounded arithmetic grammar without `eval`/`exec`; it is a scalar experiment utility, not an autodiff loss compiler wired into SFT.

These interventions are **not all applied sequentially** by default. In particular, PEFT adapter snapshots/exports do **not** capture arbitrary base-weight edits. Keep direct trials separate from SFT unless you implement and test a backend that owns/exports those mutations. Blindly combining a merge, architecture mutation and ablation can destroy a model and prevent causal attribution.

## 5. Storage and provenance

A checkpoint is written to a temporary sibling, populated with actual tensors, hashed, verified, then renamed. Its ownership marker records run ID, generation, parent, scores, backend metadata and per-file SHA-256. Hashes detect content mismatch, not malicious authors who can rewrite manifests.

An accepted weight patch is different from a checkpoint: two data-only files containing metadata, sorted flat indices and raw before/after values. Verification without Torch checks format/integrity; application additionally checks the exact target values, full target hashes and norm bound before copying. A patch does **not** bind every other model tensor, authenticate the author, certify a license assertion or prove semantic locality. Only ordinary little-endian F16/BF16/F32/F64 matrices are supported. Dense snapshots, sorting and hashing still scale with the selected whole matrix.

Pareto pruning compares macro accuracy (maximize) against actual bytes (minimize), caps the frontier and always retains protected points. Only correctly named, verified, same-run checkpoint directories are eligible; foreign, unknown, corrupt or symlinked paths are skipped. The cap can be exceeded by protected points. The CLI previews by default; trial/patch directories are not silently treated as checkpoints. The controller prunes only its private checkpoint directory after persisting a decision. Cooperative time checks cannot interrupt an in-flight GPU kernel or a hung callback.

## 6. Research-to-implementation map

| Research direction | Evidence and limits | Local surface |
| --- | --- | --- |
| LoRA / QLoRA | [Original papers](../papers/LoRA_and_QLoRA.md); resource claims depend on setup | Real PEFT SFT; GPU quantization untested |
| SEAL | [2506.10943](https://arxiv.org/abs/2506.10943): learned self-edit outer RL | Verified proposal interface + local SFT, not SEAL RL |
| Weight-space programming | [MIT post](../papers/Sparks_of_Weight_Space_Programming.md): specific repetition repair | Bounded inspection, measured reversible trials and patches; Astra not reproduced |
| Self-rewriting | [2611.21885v2](https://arxiv.org/abs/2611.21885v2): larger model gains, smaller models fail | Same-model bounded menu selection, measured trials/abstention/chain rollback and final test; 135M plateau smoke, not a reproduction or learned self-editor |
| Harness/weight co-evolution | [Evo2](https://arxiv.org/abs/2610.13025), [SIA](https://arxiv.org/abs/2605.27276) | Separate reviewable patch and adapter experiments, not joint RL |
| Evaluation and continual learning | [WeightBench](https://arxiv.org/abs/2611.18553), [continual-agent study](https://arxiv.org/abs/2610.25858) | Independent source checkouts, original/latest gates and optional final test |
| Recursive synthetic-data drift | [Nature article](https://www.nature.com/articles/s41586-024-07566-y) | Verified replay mitigation, not a universal ratio or theorem |

See the [recent review](../papers/RECENT_RESEARCH_2026.md) for primary-source dates, review scope and counterevidence.
