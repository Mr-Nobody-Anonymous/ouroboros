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

## Local verification performed for 0.3.0

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

The integration environment uses NumPy 2.4.6, PyTorch 2.5.1 (CUDA libraries installed but **no CUDA device available**), Transformers 4.57.6, PEFT 0.18.1 and Accelerate 1.15.0. Tiny random GPT-2/tokenizer and Llama models are generated locally; **no pretrained weights or benchmark corpora are downloaded**. PEFT's GPT-2 Conv1D fan-in/fan-out adjustment warnings are expected. The standalone loader also emits Transformers' `torch_dtype` deprecation notice, not a failed inference result. `pip check` passed in the ML and fresh minimal/package environments.

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

## Standalone CLI smoke workflows

These also ran outside pytest:

1. **NumPy training:** three generations produced `[promoted, rejected, rejected]`; the four-record final fixture scored macro accuracy 1.0 after selection. Checkpoint reload/evaluation also scored 1.0, and retention preview changed no files.
2. **2×2 Torch edit fixture:** one entry repaired `repair: 0.0 → 1.0` while `retention` stayed 1.0. The live model was restored; the accepted sparse safetensors payload was **216 bytes**. No-Torch CLI verification/inspection succeeded, and explicit application/reversal on a matching fixture was exact.
3. **Random local GPT-2:** generated files, listed actual matrices, inspected one using the core CLI, and tested a ten-entry edit against an unreachable canonical label. Baseline/candidate scores were both zero, so the trial correctly **rejected** it, restored the live tensor and exported no patch. This checks control flow, not math learning.

Both demos are intentionally trivial mechanical evidence, **not pretrained LLMs, a generalization study, Astra reproduction or a large math improvement**. Public color-pattern/test fixtures are not scientifically independent capability benchmarks.

## Distribution checks

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
