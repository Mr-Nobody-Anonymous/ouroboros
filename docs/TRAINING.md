# Local training and evaluation

## Installation choices

- Core: `pip install -e .` installs NumPy/PyYAML only.
- Development: `pip install -e '.[dev]'` adds pytest, Ruff and build tooling.
- Causal-LM LoRA: install a suitable PyTorch build, then `pip install -e '.[train]'`.
- CUDA 4-bit quantization: additionally `pip install -e '.[quantization]'`.
- TRL research/recipes: `preference` is an optional dependency extra, not a switch that turns the default SFT controller into DPO/GRPO.

The CPU integration test uses PyTorch 2.5.1, Transformers 4.57.6, PEFT 0.18.1 and Accelerate 1.15.0. [Tested constraints](../constraints/ml-tested.txt) record that combination; they are not a promise that all older/newer releases or GPU configurations work. The vendored PEFT/TRL trees are reference snapshots, not packages silently imported with `sys.path` modifications.

Provision a **local** causal-LM directory before running. It needs a compatible Transformers config, safetensors weights and tokenizer files. Model downloads are not performed by Ouroboros. Follow the model's license and model card. Arbitrary Hub IDs, custom remote Python code and pickle `.bin` weights are intentionally not loaded by the core backend.

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

`instruction` and `target` are required nonempty strings. Domain/source default to `general`/the input path; `verified` and `is_anchor` default to false. Unknown fields and conflicting labels are rejected rather than ignored. **Setting `verified: true` records an assertion; the software does not magically verify its truth.** Replay anchors must be explicitly verified. Do not put secrets, unauthorized personal data, licensed benchmark hold-outs or a test answer key in the training set.

`train`, `validation` and optional `anchors` are separate files. Normalized instruction overlap between train/anchors and validation is refused. Exact deduplication is not semantic leakage detection. If no verified anchors are supplied, set `cycle.anchor_ratio` to zero; a positive ratio without anchors fails instead of silently pretending to replay.

The anchor ratio is the fraction of the **final** batch, rounded up: for n new records and desired ratio r, add `ceil(n*r/(1-r))` replay records. Repetition of a tiny anchor pool does not add diversity. No research theorem establishes that 25% universally prevents collapse.

## Configuration

Edit [`examples/local_sft.json`](../examples/local_sft.json). Paths inside it resolve relative to the config file, not your shell directory.

```bash
ouroboros run --config examples/local_sft.json --check-only
ouroboros run --config examples/local_sft.json
```

`--check-only` validates schema, thresholds, data splits and counts, without loading a model. The actual run also checks model availability/architecture, context length and quantization requirements. The example data are tiny hand-authored **smoke fixtures**, not MATH-500 or a scientific training corpus.

`TuningConfig` controls LoRA rank/alpha/dropout/targets, learning rate, epochs, batch size, gradient accumulation, sequence length and gradient norm. `RecursiveCycleConfig` controls generation count, seed, replay ratio, forgetting/improvement thresholds, step/time budgets, artifact roots and free-disk minimum. All generation decisions use actual held-out predictions, not synthetic score increments.

The shared text template is deliberately simple:

```text
Instruction:
<instruction>
Response:
<completion>
```

This does not automatically apply a model-specific chat template. For a chat-tuned model, evaluate whether this format is appropriate before claiming useful adaptation. Training masks prompt/padding loss; inference decodes only the generated continuation. Overlong completion labels are refused. Training may truncate the left of a long prompt to fit; evaluation refuses overflow instead of silently changing the question.

## Evaluation and reproducibility

```bash
ouroboros evaluate \
  --checkpoint checkpoints/ACTUAL_RUN_ID/generation_0001 \
  --data examples/data/validation.json
```

For a NumPy demo checkpoint use `examples/data/demo_validation.json`, not the arithmetic data. For a relocated LLM base use `--model-path /new/local/model`.

Evaluation first verifies checkpoint file inventories and SHA-256 hashes. The built-in metric is case/whitespace-normalized **exact match**, appropriate for canonical short answers; it is not a full symbolic math grader, coding execution harness, IFEval implementation or language-quality judge. Use task-specific independent verifiers and the separate [evaluation-harness setup](INTEGRATIONS.md) for realistic benchmarks. Missing domains and nonfinite scores cannot pass the promotion gate.

`outputs/<run-id>/run.json` records configuration, baseline/candidate predictions, macro-domain decisions, training losses/steps, parent checkpoints and a data fingerprint. Checkpoint lineage links may refer to pruned ancestors; the baseline and current model are protected. Exports record SHA-256 hashes of the base safetensors and standard tokenizer/config files; adapter reload refuses a recorded provenance mismatch. Keep the original base model and its upstream revision separately: adapters alone are not standalone full models, and hashes do not recover the base weights.

### Honest experiment design

The controller reuses a validation split. That is useful for engineering, but repeated selection can overfit it. Keep a separate final test set never available to proposers/trainers, evaluate multiple seeds, compare equal token/compute budgets and measure non-target capabilities. An unchanged score is not promoted. The public literature's improvements are not local results. Time budgets are cooperative checks between steps, not hard process/VRAM quotas.

## Programmatic self-generated curricula

Inject real trusted generator and independent verifier callbacks:

```python
from ouroboros.pipeline.self_loop import RecursiveTuningLoop

# supply backend, dataset, cycle_config and tuning_config for your own experiment
loop = RecursiveTuningLoop(
    cycle_config, tuning_config, backend=backend, dataset=dataset,
    proposal_provider=propose_examples,       # (weak_domain, training_samples) -> iterable[Sample]
    proposal_verifier=independently_verify,  # Sample -> bool
)
loop.run_cycle()
```

The provider is not handed validation labels. Up to 32 proposals are considered per generation, checked for overlap/length/conflicts, then blended with verified replay. These callbacks execute as trusted application code; they are not a sandbox. Do not make an untrusted model its own unquestioned evaluator.

## Code and architecture experiments

`ouroboros propose` writes a patch and an approval=false manifest, never edits the original source. `ouroboros objective` evaluates a restricted arithmetic language, not arbitrary Python or a differentiable SFT loss. [Security guidance](../SECURITY.md) describes OS/container testing for source proposals. `ArchitectureMutator.build_layer_stacked_candidate` and reversible weight-edit utilities are explicit Python APIs, not hidden stages in every generation. Measure each candidate independently and preserve a known-good model.
