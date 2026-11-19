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
ouroboros demo --generations 3
ouroboros repos verify  # after initializing the pinned submodules
python scripts/verify_artifacts.py
python -m build
```

Optional local-LM integration tests:

```bash
pip install -e '.[train]' -c constraints/ml-tested.txt
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q -m ml
```

`pytest` runs the full suite; unittest discovery runs the smaller public-API smoke suite only. The [CI workflow](../.github/workflows/ci.yml) tests minimal Python 3.10/3.11/3.12 environments plus a separate Python 3.11 CPU local-LM job. CI results are distinct from commands run locally and should not be claimed passed until observed.

## Local verification performed

On **2026-11-19**, Python 3.11 in the Arena checkout returned **125 passed, 3 expected PEFT warnings** for the full suite. A separate fresh environment with **no Torch/Transformers/PEFT installed** returned **119 passed, 6 ML tests deselected**. Unittest compatibility discovery returned **14 passed**. Ruff lint/format, documentation/catalogue checks, source verification (**12/12**), example configuration validation and source/wheel builds also passed locally. These observations are not claims about unobserved remote CI jobs or other Python/GPU configurations.

The integration environment uses NumPy 2.4.6, PyTorch 2.5.1 (CUDA libraries installed but **no CUDA device available**), Transformers 4.57.6, PEFT 0.18.1 and Accelerate 1.15.0. A tiny GPT-2/tokenizer and a tiny Llama model are generated locally; **no pretrained weights are downloaded**. The upstream PEFT Conv1D fan-in/fan-out adjustment warning is expected for GPT-2 and not a benchmark result.

Checks exercise:

- Numerical LoRA gradients, loss measurement, frozen base tensors and exact rollback state.
- Prompt/EOS/padding label handling, context/sequence budgets, real safetensors export and reload predictions.
- Measured promotion and plateau rejection; real rollback on training and checkpoint-export failure.
- Deterministic seeded predictions/losses/parameters, per-run artifact isolation and checksum tamper detection.
- Data schema, conflicting targets, normalized split contamination, verified proposals and final-batch replay ratios.
- No substring-credit bug (`1` is not `10`), no invented baselines, finite scores, original/latest forgetting and strict improvement gates.
- Exact sparse edit counts with tied magnitudes/zero budgets, TIES/DARE/SLERP edge cases and reversible actual PyTorch edits.
- Explicit copied decoder-layer candidates with correct layer counts/cache indices and a real forward pass.
- Confidence not overriding wrong answers, explicit judge/verifier requirements and abstention on tied preferences.
- Code proposal hashes/patches without source execution and rejection of dangerous arithmetic-language constructs.
- Actual-byte Pareto retention, preview behavior and preservation of foreign, unknown, corrupt and symlinked paths.
- Bridge refusal to claim execution when missing or not explicitly authorized, source lock consistency, documentation/catalogue integrity and release/artifact tooling.

The measured demo trains a tiny color-label classifier from a baseline that predicts one label to perfect accuracy on four simple held-out prompts, then rejects plateau candidates. This is intentionally trivial **plumbing evidence**, not a 50-point improvement in real LLM mathematics or a generalization study.

## Not performed / not established

- No MATH-500, GSM8K, IFEval, GPQA or other full pretrained-model capability benchmark.
- No Astra/Qwen patch reproduction, 27B self-rewriting run, or proof of semantic weight understanding.
- No learned SEAL/EvoTIDE outer-loop RL, default DPO/GRPO training or learned minimax adversary.
- No CUDA/QLoRA performance, memory or convergence validation.
- No real EasyEdit ROME/MEMIT run, multi-model mergekit merge or full WeightBench benchmark.
- No execution of DGM's Docker/API-dependent evolution loop or the optional container policy.
- No recovery or successful upload of the historical missing 600 MB payload. Local artifact checks cannot prove remote LFS availability.
- No claim that an engineering threshold gate replaces pre-registration, multiple seeds, uncertainty, side-effect tests or a never-consulted final test set.

## Distribution checks

A built source archive was extracted without Git metadata or any upstream trees: **118 core tests passed**, one release-builder provenance test was appropriately skipped, and six ML tests were deselected. Its documentation checker explicitly reports unavailable upstream reference trees rather than confusing their absence with a broken owned file. Rebuilding a release with a commit identity still requires a Git checkout.

The built wheel was installed into a separate non-editable environment and invoked outside the source directory. The three-generation CLI demo promoted the first measured improvement, rolled back both plateaus, reloaded and evaluated the saved checkpoint, and previewed retention without deletion. No model downloads were involved.

The most important change is evidential: tests now check real state and failure behavior instead of making simulated score increases look like successful LLM training.
