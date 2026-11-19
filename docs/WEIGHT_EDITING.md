# Numeric inspection and measured weight-edit experiments

This is an explicit experimental path alongside LoRA training. It does **not** reproduce Astra, infer semantics from arbitrary weights, or silently insert weight surgery into every training generation.

## Inspect a file without loading a model

The minimal installation includes NumPy, PyYAML and safetensors; Torch/Transformers are not needed for these commands:

```bash
ouroboros weights inspect --file /path/to/model.safetensors --limit 20
ouroboros weights inspect --file /path/to/model.safetensors \
  --tensor model.layers.0.mlp.down_proj.weight --values 16
# Optional: reads the entire file to compute its checksum, so allow for its size.
ouroboros weights inspect --file /path/to/model.safetensors --checksum
ouroboros weights verify-patch --patch outputs/weight-trials/ACTUAL_ID/patch
```

Inspection reads a bounded header and, when requested, a short **flat prefix**, not a full matrix. The header limit is 16 MiB, output limit 1000 tensor records, and value limit 64 per record. F32/F64/F16/BF16 and conventional integer/bool values are supported; unsupported value encodings can still be inspected as metadata. NaN/Inf appear as explicit JSON strings. This is a numeric view—not a decoder of a neuron's meaning or a causal circuit diagnosis. Inspect individual safetensors shards; no Hub access or implicit pickle loading occurs.

## A runnable mechanical fixture

```bash
pip install -e '.[train]' -c constraints/ml-tested.txt
ouroboros weights demo --output-dir outputs/weight-trials
```

The demo edits **one entry of a 2×2 Torch classifier** with two independent fixture questions. It measures a repair and retention, exports a sparse patch when the gate passes, and restores the live model. It is **not an LLM, math benchmark or scientific generalization result**. It is deliberately small so an AI agent can inspect every number and verify the control flow.

The JSON response gives the actual `report_path` and `patch`. Keep these paths rather than assuming a particular UUID. A rejected trial is a completed experiment, not a successful model improvement; CLI exit 0 alone does not mean `passed: true`.

## A real local causal-LM trial

1. Provision a licensed local model with safetensors, config and tokenizer files. No model is downloaded by Ouroboros.
2. Copy/edit [weight_trial.json](../examples/weight_trial.json). Replace the model path, exact runtime parameter name, model license, source revision and direction provenance.
3. Replace [direction.example.json](../examples/data/direction.example.json) with a **measured** finite nonzero vector: one entry per stored matrix row. The four-entry example is a schema placeholder and does not fit arbitrary Qwen models. It is not Astra's released direction. `WeightSpaceProgrammer.contrastive_direction` can reduce caller-supplied failing/successful activations, but those activations and their causal interpretation remain your responsibility. Standard Linear and GPT-2 Conv1D store matrices in different orientations.
4. Supply independently checked canonical validation answers and retention domains. Do not tune on your final test set.

```bash
ouroboros weights trial --config examples/weight_trial.json --check-only
# Explicitly loads a model to discover runtime matrix names; does not edit it.
ouroboros weights trial --config examples/weight_trial.json --list-parameters
# Explicitly runs baseline and candidate inference and writes an experiment report.
ouroboros weights trial --config examples/weight_trial.json
```

Config paths resolve relative to the config file. `--check-only` checks schema, the bounded direction JSON and validation data; it does **not** establish that a model exists or that its target shape matches. The actual trial uses the same offline safetensors loader/greedy text template as SFT, without inserting LoRA. Quantized models and non-floating/non-matrix parameters are refused. Nothing performs SVD-based semantic diagnosis or automatic layer selection.

### Trial contract

- Snapshot the selected parameter and module training flags. Use paired Torch seeds for baseline/candidate inference and restore caller Torch RNG state.
- Compute a directional ablation and retain at most `floor(target_numel * edited_fraction)` changed entries of the **selected matrix**, not a fraction of the whole model. Ties never enlarge the budget. Preserve F64 precision; F16/BF16 weights use at least F32 intermediate arithmetic and are rounded back once.
- Measure the **rounded** actual edit and enforce its relative Frobenius-norm and sparse-file byte budgets. A zero/rounded-away edit is rejected.
- Require measured macro-domain accuracy gain and per-domain retention on the fixed validation records. The metric is normalized exact match; it is not a full mathematical or language-quality grader.
- **Always restore the live parameter and training flags**, including acceptance, callback error, cancellation and disk/export failure. Accepted trials export a proposal patch, not a permanently modified model.
- Export sparse indices plus old/new raw values in safetensors, bound to full before/after **target-tensor** hashes, shape, dtype, count/byte/norm budgets, license assertion and provenance. No pickle or generated Python is loaded.

`trial.json` records measured predictions, per-domain counts, scores, the decision and restoration status. `patch/patch.json` and `patch/delta.safetensors` exist only for accepted exported candidates. Reports/patches are ignored local outputs, not silently uploaded data.

## Explicit application, evaluation and reversal

```python
from ouroboros.core import apply_weight_patch, verify_weight_patch

verify_weight_patch(patch_directory)              # no Torch/model needed
apply_weight_patch(model, patch_directory)       # explicit policy decision: modifies the matching tensor
# Independently evaluate model here.
apply_weight_patch(model, patch_directory, reverse=True)
```

Application checks file checksums, finite old/new values, unique/in-range indices, the complete expected input tensor, the declared relative-norm bound and the expected result **before copying** into the live parameter. Reapplying a patch to an already patched or different tensor fails. Reverse application needs the exact patched tensor; it is not approximate subtraction. Data-only verification alone cannot establish that the supplied model has the right full-model provenance.

For a local model, score an exported patch on a separate file and undo it automatically:

```bash
ouroboros weights evaluate-patch --config examples/weight_trial.json \
  --patch outputs/weight-trials/ACTUAL_ID/patch --data examples/data/test.json
```

This loads a fresh base model, applies the patch for scoring, reverses it in `finally`, and never writes a full model. Use genuinely independent data; the CLI cannot prove semantic independence from outside experiments. A patch is not an adapter, standalone model, or proof of locality. **LoRA-only checkpoints do not capture arbitrary base-weight edits.** Do not apply base edits inside the SFT controller and assume its adapter snapshots/exports will preserve or roll them back. Keep these experiments separate unless you implement/test the corresponding backend state/export contract.

## Limits and sharing

- Dense copies, sorting and norm computations still scale with the selected whole matrix. A small serialized patch does not imply constant memory or compute. Run under external quotas; there is no hard OS/VRAM/time sandbox.
- Only little-endian hosts and ordinary F16/BF16/F32/F64 matrices are currently supported. CPU tests do not establish GPU/quantization behavior.
- A supplied `predict` callback is trusted, deterministic and read-only. Labels are not passed to it, but arbitrary callback writes to other parameters, buffers or files are not isolated. Do not run during an in-flight autograd/optimizer step.
- Tensor hashes detect content mismatch, not author identity, base-model licensing or effects of other weights. Weight tying can make one tensor serve several roles. A tiny edit may have broad effects.
- Preserve baseline/current models and evaluate non-target tasks and a final test set. Repeated validation selection can overfit; bounds and replay are mitigations, not no-forgetting theorems.
- Patches inherit relevant base-model rights. The example's license/provenance placeholders must be replaced; nonempty text is an assertion, **not automatic legal verification**. Keep both patch files together if packaging/registering an approved artifact. See [storage and releases](STORAGE_AND_RELEASES.md); no model weights or historic missing 600 MB are included by default.

See [architecture](SELF_EVOLVING_ARCHITECTURE.md), [testing](TESTING.md), and [the corrected Astra research note](../papers/Sparks_of_Weight_Space_Programming.md).
