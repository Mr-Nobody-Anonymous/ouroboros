# Verification and what the results do not prove

## Commands

```bash
pip install -e '.[dev]'
python -m pytest -q -m 'not ml'
python -m unittest discover -s tests -v
ruff check ouroboros tests scripts
ruff format --check ouroboros tests scripts
python scripts/check_docs.py
ouroboros run --config examples/local_sft.json --check-only
ouroboros weights trial --config examples/weight_trial.json --check-only
ouroboros demo --generations 3
ouroboros repos verify  # after initializing the pinned submodules
python scripts/verify_artifacts.py
python -m pip check
python -m build
```

Optional Torch/direct-edit and local-LM tests:

```bash
pip install -e '.[train]' -c constraints/ml-tested.txt
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
# Or select only optional tests with: python -m pytest -q -m ml
ouroboros weights demo
```

`pytest` runs the full suite; unittest discovery runs the smaller public-API smoke suite. The [CI workflow](../.github/workflows/ci.yml) tests minimal Python 3.10/3.11/3.12 plus a separate Python 3.11 CPU ML job. Remote CI conclusions are distinct from local observations and must be checked before being claimed.

## Local verification performed for 0.4 development

On **2026-11-19**, the active implementation returned **249 passed** (3 expected PEFT warnings) in Python 3.11.2 with the tested ML stack plus MCP 1.30.0 and GGUF 0.19.0. A **fresh installer-created core+MCP environment with no Torch/Transformers/PEFT/GGUF returned 212 passed, 37 ML tests deselected**. Unittest compatibility: 14 passed. Ruff lint/format: 58 Python files; owned Markdown/research checks: 32 documents / 33 references. Dependency consistency passed. These counts are new runs, not reused 0.3 evidence.

New regressions exercise packaged `init`/check-only without heavy imports, installer plans, config/resource/entry/norm/model-byte limits, held-out split separation, scoped paths/symlinks/shard-index traversal, explicit-download commit/size/inventory gates and interrupted-download cleanup. Mechanical Torch fixtures cover multi-patch acceptance followed by exact reverse-order cleanup, original-baseline retention across rounds, scalar buffers, module flags/RNG, abstention, no-op rejection, candidate errors, cancellation, final-test failure and failed reversal. A real random local Llama/tokenizer tests chat rendering, unique single-token constrained selection and an actual same-model edit; it remains a **random fixture**, not pretrained capability evidence.

MCP tests use the official SDK and real stdio protocol initialization/list/calls. They check six read-only default tools, absence of execution by default, explicit opt-in+allowlist/hash enforcement, fixed offline-Hub child arguments, path/report/inspection bounds, serialization, noisy-child timeout cleanup and cancellation. No model download occurs during ordinary pytest.

### Actual pretrained CPU run, also through MCP

A pinned/hash-checked **Q4_1-derived SmolLM2-135M-Instruct (134,515,008 parameters)** was actually cloned/provisioned from GitHub and converted to float32 safetensors, not initialized randomly and not restored to original full precision. The standalone CLI smoke passed. A separate enabled **real stdio MCP** session initialized, discovered seven tools, checked config/weights, ran the same pretrained loop in its offline-Hub child, and read its report back successfully.

| Observation | Recorded local result |
| --- | --- |
| Model-selected target | `model.layers.28.mlp.down_proj.weight`, shape `576 × 1536`, row 17 |
| Selected strengths | `0.1`, then `0.25` (the same model's menu-label argmax) |
| Actual entries changed per trial | **884**, **884** (each within a 0.1% one-matrix budget) |
| Relative norm changes | About **0.5000%**, **1.2501%** (both under the 2% default cap) |
| Four-example validation macro exact match | **0.0 → 0.0**, then **0.0 → 0.0** |
| Decisions / exported patches | **Rejected / rejected; zero accepted patches** |
| Final-test calls | Once after selection; two public smoke examples, not used for promotion |
| Complete model restoration | Before/after state SHA-256 equal; `base_restored: true` |
| Source model/tokenizer files | Rehashed unchanged; `source_files_unchanged: true` |

Full [pretrained report](evidence/smollm2-self-edit-local.json), [MCP session summary](evidence/mcp-local-smoke.json), [source/conversion provenance and reproduction commands](SELF_EDITING.md). The report includes actual predictions and conditional menu probabilities. The initialized server's implementation version in the protocol summary is the SDK version; `status.version` reports Ouroboros's application version.

The strict metric penalized explanatory/truncated continuations: `The capital of France is Paris.` did not exactly equal `Paris`. We did **not** relabel targets, loosen the gate or claim a gain. Repeated runs of these public fixtures are plumbing checks, not independent capability evaluation, confidence intervals, semantic weight understanding or a research reproduction. No MATH/GSM8K/IFEval dataset was run.

### Problems encountered and fixed

- A generic fast tokenizer returned `token_type_ids` unsupported by Llama generation; prompt encoding now explicitly omits them for inference/proposals, with real Llama regression coverage.
- A permissive model-ID pattern admitted `../evil`; validation now rejects it before the mocked network boundary.
- Scalar state buffers need flattening before byte views; complete state hashing now includes them safely without changing existing matrix patch hashes.
- Original-generation retention is now passed into each successive weight trial, preventing threshold-sized incremental drift.
- Timeout cleanup now drains pipes as well as waiting/terminating, with a noisy sleeping child regression, concurrency rejection and cancellation tests.
- Failed/interrupted final tests and failed chain reversal are recorded as failures rather than silently reported completed. A transient missing proposal-helper import was caught by Ruff/real MCP execution and fixed before publishing.

The original Hub provisioning alternative has **mocked boundary tests only** in this restricted environment. Source checkouts from the prior phase are absent/empty after workspace restoration; documentation validation explicitly reports all seven omitted submodules, rather than claiming a new 12/12 verification. The pinned smoke-model source was independently cloned and hash-verified in this phase.

### Distribution and installation

The one-command installer ran into a **fresh core+MCP venv** without Torch; its suite/doctor/dependency checks passed as above. A non-editable 0.4 wheel was then installed in another **no-Torch, no-MCP** venv and invoked outside the source checkout. Packaged `init`, self-edit check-only, the measured three-generation NumPy demo (`[accepted, rejected, rejected]`), final-test macro 1.0, checkpoint reload macro 1.0, actionable missing-MCP errors and `ouroboros-mcp --help` passed. A test-driver initially looked for an obsolete report key; reading the actual `history/passed` schema corrected that driver without changing product code or its data.

An extracted 0.4 sdist with no Git/upstream trees passed **203 tests**, with **2 expected skips** (MCP SDK absent and Git-provenance builder) and **37 ML tests deselected**. Its docs checker passed. Wheel resources/source bytes were verified; sdist inventory contains the emblem, small evidence JSON, installer and pretrained smoke script. Neither archive contains weights, GGUF, caches or upstream trees. `pip check` passed in the ML, fresh core+MCP and minimal wheel environments. These are build/install checks, not a PyPI or release upload.

### GitHub verification

The updated workflow separates core Python 3.10/3.11/3.12, offline CPU/random-LM, no-Torch MCP protocol, and an explicit pinned pretrained-model smoke. The pretrained job requires real nonzero same-model trials, full restoration/source checks and final-test control flow; a plateau/rejection is a legitimate test outcome. Only JSON evidence is uploaded, never pretrained weights.

The actual implementation commit **`2cade5452f36fc742f7815bf42e26e956888f679`** passed **all six jobs on both** [branch run 44679460633](https://github.com/Mr-Nobody-Anonymous/ouroboros/actions/runs/44679460633) and [PR run 44679466730](https://github.com/Mr-Nobody-Anonymous/ouroboros/actions/runs/44679466730). This includes the real pinned pretrained script, not just random fixtures. Historical 0.3 CI is not reused as evidence for these features. The follow-up CI configuration uses native Node 24 actions (checkout v5 / setup-python v6 / upload-artifact v6) instead of relying on forced migration of Node 20 actions. Check [PR #1](https://github.com/Mr-Nobody-Anonymous/ouroboros/pull/1) for the latest head's runs.

GitHub's artifact API confirmed the initial branch run's `pretrained-self-edit-evidence` ZIP (**5,962 bytes**, artifact `14603727833`, reported SHA-256 `06d7f7ce2b7fe384e551069ee7822a8668b2519ba3631c9492cacf5371194d7c`). An independent download in this sandbox failed at GitHub's redirected Azure host (EOF); **its payload was not locally verified** and no GitHub report is copied into this repository as if downloaded. The workflow also emits a compact measured JSON notice/job summary, readable through GitHub's check-annotations API without that artifact host. Only validated counts/scores/decisions/restoration hashes are included—no predictions, private paths or environment secrets. This reporting path has a dedicated no-network regression test. Normal GitHub users can also retrieve the full JSON artifact with `gh run download RUN_ID --name pretrained-self-edit-evidence`.

## Historical local verification performed for 0.3.0

On **2026-11-19**, Python 3.11.2 in the Arena checkout returned:

| Check | Observed result |
| --- | --- |
| Full offline suite | **195 passed**, 3 expected PEFT warnings |
| Fresh no-Torch/Transformers/PEFT environment | **168 passed**, 27 ML tests deselected |
| Unittest compatibility discovery | **14 passed** |
| Ruff lint/format | Passed (47 Python files formatted) |
| Owned Markdown / research metadata | **30 documents / 33 references**, consistent |
| Source inventory | **12/12 verified**, seven submodules + five snapshots |
| Artifact registry | **0 files / 0 bytes**; no remote-LFS availability claim |
| Config checks / workflow YAML / shell syntax / dependency consistency | Passed |

The integration environment uses NumPy 2.4.6, PyTorch 2.5.1 (CUDA libraries installed but **no CUDA device available**), Transformers 4.57.6, PEFT 0.18.1 and Accelerate 1.15.0. In that 0.3 phase, tiny random GPT-2/tokenizer and Llama models were generated locally; **no pretrained weights or benchmark corpora were downloaded for those checks**. PEFT's GPT-2 Conv1D fan-in/fan-out adjustment warnings are expected. The standalone loader also emits Transformers' `torch_dtype` deprecation notice, not a failed inference result. `pip check` passed in the ML and fresh minimal/package environments.

Checks exercise:

- Numerical LoRA gradients, measured loss, frozen base tensors and exact rollback state.
- Prompt/EOS/padding label handling, context/sequence budgets, real safetensors export and reload predictions.
- Measured promotion/plateau rejection and rollback on curriculum, training, evaluation and checkpoint-export failures, with durable stage/training evidence.
- Seeded predictions/losses/parameters, per-run isolation and checksum tamper detection.
- Schema, conflicting targets, normalized split contamination, verified proposals and final-batch replay ratios.
- Disjoint optional final test: no baseline/proposer/training/promotion use; one post-selection call; budget-limited selection, absent/failed/skipped/interrupted states and nonzero CLI failure behavior.
- Complete prediction-vector type/cardinality checks before scoring; `1` not matching `10`; no invented baselines; original/latest retention and strict positive improvement.
- Bounded numeric inspection with no Torch, BF16/nonfinite decoding, oversized/invalid headers and offset rejection, plus short-prefix reads of a synthetic sparse file.
- Accepted/rejected Torch trials across F16/F32/F64/BF16, exact entry/norm/payload-byte gates, zero edits, plateau rejection, cancellation, paired/restored Torch RNG, heterogeneous training flags and export failure.
- Exact patch apply/reverse, wrong-state/target refusal, indices/shape/dtype/checksum/norm validation, rehashed inconsistent payloads, extra files and symlinks. Unedited F64 values and **signed-zero bytes** remain intact, including Fortran-layout NumPy input.
- Real shared bare-loader inference, local runtime parameter listing and a measured random-GPT-2 edit trial that correctly rejects zero improvement and restores the target.
- Exact sparse budgets with ties/zero, TIES/DARE/SLERP edge cases, copied decoder-layer candidates with cache indices and a real forward pass.
- Confidence not overriding wrong answers, explicit judge/verifier requirements and abstention on tied preferences.
- Code proposal hashes without source execution; rejection of dangerous arithmetic-language constructs.
- Actual-byte Pareto retention/preview and preservation of foreign, unknown, corrupt and symlinked paths; bridge refusal to invent execution.

## Historical 0.3 standalone CLI smoke workflows

These also ran outside pytest:

1. **NumPy training:** three generations produced `[promoted, rejected, rejected]`; the four-record final fixture scored macro accuracy 1.0 after selection. Checkpoint reload/evaluation also scored 1.0, and retention preview changed no files.
2. **2×2 Torch edit fixture:** one entry repaired `repair: 0.0 → 1.0` while `retention` stayed 1.0. The live model was restored; the accepted sparse safetensors payload was **216 bytes**. No-Torch CLI verification/inspection succeeded, and explicit application/reversal on a matching fixture was exact.
3. **Random local GPT-2:** generated files, listed actual matrices, inspected one using the core CLI, and tested a ten-entry edit against an unreachable canonical label. Baseline/candidate scores were both zero, so the trial correctly **rejected** it, restored the live tensor and exported no patch. This checks control flow, not math learning.

Both demos are intentionally trivial mechanical evidence, **not pretrained LLMs, a generalization study, Astra reproduction or a large math improvement**. Public color-pattern/test fixtures are not scientifically independent capability benchmarks.

## Historical 0.3 distribution checks

The current-code Python source distribution was extracted without Git metadata or upstream trees: **167 core tests passed**, one Git-provenance release-builder test was correctly skipped, and 27 ML tests were deselected. Documentation validation passed and explicitly listed omitted upstream references instead of treating them as broken owned files. Rebuilding a release with a commit identity requires a Git checkout.

The wheel's Python modules were checked against source bytes, installed **non-editably** into a separate no-Torch environment and invoked outside the source tree. The three-generation demo, final-test result, checkpoint reload/evaluation, config checks, data-only patch verification/inspection and signed-zero regression smoke passed. Wheel/sdist inventories exclude model payloads and upstream trees. These checks are distinct from the deterministic Git-provenance core archive's checksum/content manifest and from actual remote asset publication. Builds are not a PyPI upload.

## Not performed / not established

- No MATH-500, GSM8K, IFEval, GPQA or other full pretrained-model capability benchmark.
- No Astra/Qwen patch reproduction, 27B self-rewriting run or proof of semantic weight understanding.
- No learned SEAL/EvoTIDE outer-loop RL, default DPO/GRPO training or learned minimax adversary.
- No CUDA/QLoRA/direct-GPU performance, memory or convergence validation; direct quantized edits are refused.
- No actual EasyEdit ROME/MEMIT run, multi-model mergekit merge or full WeightBench benchmark.
- No execution of DGM's Docker/API-dependent loop or the optional container policy.
- No isolation of arbitrary trusted callback writes to other tensors/files; trial ownership is limited to the selected tensor, flags and Torch RNG.
- No proof that a small serialized patch is cheap in dense RAM/compute, or that target hashes establish full-model identity, author authenticity or licensing rights.
- No recovery/upload of a missing historical 600 MB payload. Local artifact checks cannot prove remote LFS availability.
- No claim that an engineering gate replaces pre-registration, multiple seeds, uncertainty, side-effect tests or genuinely unseen final data. Repeated retuning on final results would invalidate its intended role.

The important change is evidential: tests exercise real state and failure behavior instead of making simulated score increases look like successful LLM training.
