# Local training and evaluation

## Installation choices

- Core: `pip install -e .` installs NumPy, PyYAML and safetensors. Inspection/patch-format verification require no Torch.
- Development: `pip install -e '.[dev]'` adds pytest, Ruff and build tooling.
- Causal-LM LoRA or local weight-edit trials: install a suitable PyTorch build, then `pip install -e '.[train]'`.
- CUDA 4-bit training: additionally `pip install -e '.[quantization]'`. Direct weight trials refuse quantized models.
- TRL research/recipes: `preference` is an optional dependency extra, not a switch that turns the default SFT controller into DPO/GRPO.

The CPU integration tests use PyTorch 2.5.1, Transformers 4.57.6, PEFT 0.18.1 and Accelerate 1.15.0. [Tested constraints](../constraints/ml-tested.txt) record that combination; they are not a promise that all older/newer releases or GPU configurations work. Vendored PEFT/TRL trees are reference snapshots, not packages silently imported with `sys.path` modifications.

Provision a **local** causal-LM directory before running. It needs a compatible Transformers config, safetensors weights and tokenizer files. Downloads are not performed by Ouroboros. Follow the model's license/model card. Arbitrary Hub IDs, custom remote Python code and pickle `.bin` weights are intentionally not loaded by the core backend.

## Dataset schema

Use a JSON array or newline-delimited JSON records. Small JSONL examples remain plain Git text, not LFS pointers.

```json
{
  "instruction": "What is 2 + 2? Reply with the number only.",
  "target": "4",
  "domain": "math_logic",
  "source": "human-authored-example",
  "verified": true,
  "is_anchor": false
}
```

`instruction` and `target` are required nonempty strings. Domain/source default to `general`/the input path; `verified` and `is_anchor` default to false. Unknown fields and conflicting labels are rejected. **Setting `verified: true` records an assertion; the software does not verify its truth.** Replay anchors must be explicitly verified. Do not put secrets, unauthorized personal data, licensed benchmark hold-outs or a test answer key in training.

`data.train`, `data.validation`, optional `data.anchors` and optional `data.test` name separate files. Normalized instruction overlap between train/anchors and validation is refused; test must also be disjoint from every other split. Deduplication is not semantic leakage detection. An omitted/empty test split remains valid and is reported `not_configured`; it preserves the legacy fingerprint payload. A nonempty test split is included in the dataset fingerprint and never used for training/proposals/promotion.

If no verified anchors are supplied, set `cycle.anchor_ratio` to zero; a positive ratio without anchors fails instead of pretending to replay. The ratio is the fraction of the **final** batch, rounded up: for n new records and desired ratio r, add `ceil(n*r/(1-r))` replay records. Repetition of a tiny pool does not add diversity. No theorem establishes that 25% universally prevents collapse.

## Configuration

Edit [`examples/local_sft.json`](../examples/local_sft.json). Paths inside it resolve relative to the config file, not your shell directory.

```bash
ouroboros run --config examples/local_sft.json --check-only
ouroboros run --config examples/local_sft.json
```

`--check-only` validates schema, thresholds, data splits, counts and fingerprint without loading a model. The actual run also checks model availability/architecture, context length and quantization requirements. The example includes two final-test records; all example data are tiny hand-authored **smoke fixtures**, not MATH-500 or a scientific training corpus.

`TuningConfig` controls LoRA rank/alpha/dropout/targets, learning rate, epochs, batch size, gradient accumulation, sequence length and gradient norm. `RecursiveCycleConfig` controls generation count, seed, replay ratio, forgetting/improvement thresholds, step/time budgets, artifact roots and free-disk minimum. Generation decisions use actual validation predictions, not synthetic score increments. Positive macro-domain improvement and original/latest per-domain retention are required; a plateau is rejected.

The shared text template is deliberately simple:

```text
Instruction:
<instruction>
Response:
<completion>
```

`LocalCausalLM` provides frozen offline loading/greedy inference; the SFT subclass adds LoRA with this same template. This does not apply a model-specific chat template. Evaluate whether this format is appropriate before claiming useful adaptation. Training masks prompt/padding loss; inference decodes only the generated continuation. Overlong completion labels are refused. Training may truncate the left of a long prompt to fit; evaluation refuses overflow rather than changing the question silently.

## Evaluation, final test and reproducibility

```bash
ouroboros evaluate \
  --checkpoint checkpoints/ACTUAL_RUN_ID/generation_0001 \
  --data examples/data/validation.json
```

For a NumPy demo checkpoint use `examples/data/demo_validation.json`, not the arithmetic data. For a relocated LLM base use `--model-path /new/local/model`.

Evaluation first verifies checkpoint inventories and SHA-256 hashes, then validates the complete prediction sequence's count and string types before scoring. The metric is case/whitespace-normalized **exact match**, appropriate for canonical short answers, not a symbolic math grader, coding execution harness, IFEval implementation or language-quality judge. Results include domain scores/counts, macro accuracy and predictions. Use task-specific independent verifiers and the separate [evaluation-harness setup](INTEGRATIONS.md) for realistic benchmarks. Missing domains and nonfinite scores cannot pass the promotion gate.

`outputs/<run-id>/run.json` records configuration, baseline/candidate predictions, selection decisions, completed training metrics, stage reached, parent checkpoints and dataset fingerprint. Checkpoint lineage links may refer to pruned ancestors; baseline/current models are protected. Curriculum/training/evaluation/export failure restores the backend's owned adapter/RNG state and records a terminal generation error. Cancellation is persisted and reraised. These are not guarantees about arbitrary callback file writes or other unowned state.

With `data.test` configured, `run.json.final_test` records **one evaluation after completed or budget-limited selection**, with `used_for_promotion: false`. No baseline test score is taken. Test prompts are blocked from generated curricula; the provider sees only training examples and the weak-domain name. A final-test failure sets run status `final_test_failed` and CLI exit 1 without undoing an accepted model; a failed selection skips test evaluation. When no test was supplied the report says so. The NumPy demo also has a four-record final fixture, not a scientific benchmark.

Exports record SHA-256 hashes of base safetensors and standard tokenizer/config files; adapter reload refuses a recorded provenance mismatch. Keep the original base and its upstream revision separately: adapters are not standalone models, and hashes do not recover weights. Optimizers reset each generation; full mid-step training resume is not implemented.

### Honest experiment design

Repeated selection can overfit validation. Keep genuinely independent final tests, do not retune on their results, evaluate multiple seeds, compare equal token/compute budgets and measure non-target capabilities. Same-process callbacks are trusted and can bypass API argument separation if given filesystem access. The public literature's improvements are not local results. Step/time budgets are cooperative, not hard process/VRAM quotas; final inference can outlast the selection budget.

## Programmatic self-generated curricula

Inject real trusted generator and independent verifier callbacks:

```python
from ouroboros.pipeline.self_loop import RecursiveTuningLoop

# Supply backend, dataset, cycle_config and tuning_config for your own experiment.
loop = RecursiveTuningLoop(
    cycle_config, tuning_config, backend=backend, dataset=dataset,
    proposal_provider=propose_examples,       # (weak_domain, training_samples) -> iterable[Sample]
    proposal_verifier=independently_verify,  # Sample -> bool
)
loop.run_cycle()
```

Up to 32 proposals are considered per generation, checked for validation/test overlap, length and conflicting labels, then blended with verified replay. Proposer/verifier errors are recorded at the curriculum stage and terminate that run after owned-state rollback. These callbacks execute as trusted application code, not in a sandbox. Do not make an untrusted model its own unquestioned evaluator.

## Code, weight and architecture experiments

`ouroboros propose` writes a patch and an approval=false manifest, never edits the original source. `ouroboros objective` evaluates a restricted arithmetic language, not Python or a differentiable SFT loss. [Security guidance](../SECURITY.md) describes OS/container testing for source proposals.

`ArchitectureMutator.build_layer_stacked_candidate` and the [measured weight-edit commands/APIs](WEIGHT_EDITING.md) are explicit experiments, not hidden stages in every generation. Direct trials always restore their selected parameter and export accepted patches for separate apply/evaluate/reverse decisions. **Adapter-only snapshots/exports do not capture arbitrary base-weight edits.** Keep direct trials separate unless you implement/test the appropriate backend state/export contract. Measure each intervention independently and preserve a known-good model.
