# A small LLM choosing edits to its own weights

Ouroboros 0.4 adds a **bounded, same-model experiment**, not an unrestricted self-modifying program. The very `LocalCausalLM.model` object whose next-token logits choose an action is the object edited and evaluated. No external frontier-model proposer, generated Python, model-written promotion rule or self-assigned reward is involved.

## Install and run

From the current development branch (see [README](../README.md)):

```bash
python scripts/install.py --profile llm
source .venv/bin/activate  # Windows: .venv\Scripts\activate
ouroboros doctor
ouroboros init ouroboros-lab
ouroboros models fetch-smoke --directory ouroboros-lab/models/smollm2
ouroboros self-edit --config ouroboros-lab/self_edit.json --check-only
ouroboros self-edit --config ouroboros-lab/self_edit.json
```

Use a **new** project/model destination; existing paths are refused, not overwritten. `init` works from an installed wheel too: its four small JSON resources are packaged, not read from a source-checkout-only `examples/` path. Config paths resolve relative to the config file, not the current directory. The install script supports `--profile core`, `--profile train`, `--dev`, a custom `--venv`, and `--dry-run`.

The tested ML constraints target Python 3.10–3.12. On a newer interpreter choose a supported Python or deliberately opt into `--unconstrained`; that combination has not been validated here. PyTorch packages can be several GB depending on the wheel, and inference/conversion need a few GB of free RAM. A small serialized patch is not a small dense-memory operation. A CPU-specific PyTorch wheel can reduce installation size; follow PyTorch's own platform instructions rather than installing random wheels.

### Two explicit model sources

**GitHub smoke source (run in this environment):** `models fetch-smoke` fetches data from a pinned Git repository, verifies the GGUF size/SHA-256, and converts it with installed Transformers/GGUF into ordinary float32 safetensors. It does **not** install or execute the cloned LLM plugin.

| Provenance | Value |
| --- | --- |
| Original pretrained model | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| Distributed derivative | `QuantFactory/SmolLM2-135M-Instruct-GGUF`, Q4_1 |
| Quantizer revision linked by the distributor | `ab810cf68114990406fdf996510dd3d3c6adbdf5` |
| GitHub source, actually cloned/read | [simonw/llm-smollm2](https://github.com/simonw/llm-smollm2/tree/afed0963b3a8bcca17c83f451a161eb2edbc59fe) |
| Pinned Git commit | `afed0963b3a8bcca17c83f451a161eb2edbc59fe` |
| GGUF bytes | `98,362,432` |
| GGUF SHA-256 | `b179c9523d0e6a0f98a330c7562b682750a6f8c8c15e5bc70ea373728110db53` |
| License assertion / copied upstream license | Apache-2.0 |
| Converted parameter count | `134,515,008` |
| Local converted model/tokenizer payload | `541,619,017` bytes with the tested conversion stack |

Dequantizing does **not** recover precision lost during Q4_1 quantization. These are not the original full-precision Hub weights. Git checkout data, license, metadata and tensor conversion were inspected/adapted; upstream plugin/llama.cpp inference is not used. The clone is temporary during normal provisioning; only data, license and a provenance receipt remain. Our earlier research-source inventory is unchanged.

**Original Hub checkpoint (alternative, not downloaded in this restricted environment):**

```bash
# Use a different/new destination if you already provisioned the smoke model.
ouroboros models fetch --repo HuggingFaceTB/SmolLM2-135M-Instruct \
  --directory ouroboros-lab/models/original-smollm2 --max-bytes 1073741824
# Then update model.model_name_or_path in self_edit.json to models/original-smollm2.
```

The explicit Hub command resolves a revision to a commit, checks declared file sizes before download, excludes pickle/Python/nested files and saves a receipt. It has mocked download-boundary tests, **not a live Hub download claim**. Models requiring custom remote code or nonstandard layouts are not supported by this convenience path.

Both sources write `ouroboros_model.json`, with revision/license assertions and SHA-256 hashes of every supported model/tokenizer file. Verify offline with:

```bash
ouroboros models verify --directory ouroboros-lab/models/smollm2
```

A receipt detects content changes; it is not a signature, independent proof of provenance, license approval or semantic safety certificate. Review model/dataset rights before redistribution. No model is silently downloaded during loading, check-only, inference or MCP tool calls.

## The experiment, precisely

1. `DatasetBundle` checks normalized prompt disjointness across practice/train, validation and final test. Every validation domain must appear in practice/train. Public packaged tasks are smoke fixtures, **not a statistically independent benchmark**.
2. Preflight bounds safetensors bytes and stored tensor elements before loading. It refuses symlinked model files and shard-index traversal. Trusted model/config/library files and an honest, nonconcurrent filesystem remain assumptions; this is not a general malicious-model resource sandbox.
3. The host offers at most four actions: two ablation strengths (`0.1`, `0.25`) on each of the last two eligible small MLP output matrices. Each action uses a unit coordinate at the matrix's largest-norm output row. If output projections are unavailable, eligible MLP matrices are used. **This is a transparent numerical heuristic, not semantic or causal circuit discovery.**
4. The same target model scores unique single-token labels with its own next-token logits. Argmax over that restricted menu chooses an action. The reported probabilities are **conditional on the menu**, not calibrated confidence in improvement or unrestricted generation probabilities. Position bias can dominate; choosing an action does not show that the model understands the weights.
5. The proposer receives at most two practice examples/answers, its practice predictions, the menu and earlier action/decision summaries. It never receives validation/final-test questions or labels. The trusted host alone evaluates them. Callbacks are not an OS isolation boundary.
6. A trial ablates the chosen coordinate and keeps at most `floor(matrix_entries * edited_fraction)` entries. It enforces actual rounded change count, relative Frobenius norm and serialized-patch-byte limits. This fraction is of **one matrix**, not the whole model.
7. Candidates need strictly positive macro-domain exact-match gain over the current selected state and per-domain retention against **both the original and latest selected** state. Equality is rejected. The gate is deterministic engineering policy, not a significance test.
8. Every trial restores its live matrix. An accepted patch is then explicitly applied in memory by the controller; the next round proposes from that selected model. Rejected actions are not repeated. After selection the final-test split is scored once, without influencing promotion.
9. In `finally`, accepted patches are undone **in reverse order**, heterogeneous module training flags and caller Torch RNG are restored, and a full state-dict hash is compared to the original. This includes scalar buffers. Configured runs also rehash the original model/tokenizer files. No full-model export overwrites the source.

### Defaults and boundaries

The packaged config requests two rounds, two CPU threads, a cooperative 240-second selection budget, 0.1% of one matrix, 2% relative norm and zero allowed per-domain forgetting. Host policy caps rounds at four, edited fraction at 1%, relative norm at 10%, model parameters/bytes and eligible matrix size. Dense copies/sorting/hashing and in-flight inference can exceed cooperative time expectations; use external quotas.

`require_attempt: true` in the **smoke config** deliberately removes abstention so that the mechanics are exercised. The model still chooses **which** allowed edit. Set it to `false` to offer `A = abstain` (the policy/API default). Neither setting guarantees a useful edit. No attempt is promoted merely because it was requested.

Accepted patch chains are data, not complete checkpoints. Keep them in order; each patch expects its exact predecessor tensor state. Patch hashes bind the target tensor only; the top-level full-state hashes provide audit evidence, not signed enforcement on another supplied model. Keep an independent known-good base and the original license. See [the patch API contract](WEIGHT_EDITING.md).

## Observed pretrained run

The [checked-in local report](evidence/smollm2-self-edit-local.json) records an actual CPU run, separate from random-LM unit tests. Source model: the pinned Q4_1-derived 134,515,008-parameter SmolLM2 above. The two allowed rounds performed nonzero changes, evaluated real greedy continuations and rejected both candidates on a plateau. The live full-state fingerprint restored and original model files were unchanged. **Zero accepted patches; no capability gain.**

The metric is whitespace/case-normalized **exact match**, with four validation examples (two arithmetic, two retention) and two final-test examples. It does not remove punctuation, extract numbers or give substring credit. With a 12-token continuation budget the model produced explanatory/truncated answers. For example, `The capital of France is Paris.` does not exactly equal `Paris`. The observed validation macro score was **0.0 before and after**; this is not evidence that the model knows no facts. It is also not a reason to quietly relabel evaluation data or change the metric to manufacture a gain.

The prompt/model setup was not tuned on these final answers. Repeated local/CI runs of the same **public smoke fixture** test reproducibility/mechanics only; they do not turn the fixture into unseen scientific evaluation. Use new, licensed, independently designed data and multiple seeds for capability research. Neither Astra nor the 27B self-rewriting paper is reproduced here.

Reproduce the end-to-end contract (a plateau is a valid success for this test):

```bash
# NEW workdir; reuse an already provisioned, receipt-verified smoke model:
python scripts/run_pretrained_smoke.py --workdir outputs/pretrained-check \
  --model-dir ouroboros-lab/models/smollm2
# Or explicitly allow GitHub provisioning in a new workdir:
python scripts/run_pretrained_smoke.py --workdir outputs/pretrained-fresh --fetch
```

The script fails if no real same-model nonzero trials occur, restoration/source-file checks fail, or final-test control flow is missing. It does **not** require an accuracy gain. [GitHub Actions](../.github/workflows/ci.yml) runs the same pinned smoke and uploads small JSON evidence, never model weights.

## Troubleshooting

| Symptom | What to do |
| --- | --- |
| Missing Torch/Transformers/GGUF | Run the `llm` installer profile; inspect `ouroboros doctor`. |
| Destination already exists | Verify/reuse it or select a new path. Do not delete arbitrary model directories to get past the check. |
| Hub unavailable / restricted network | Use the explicit GitHub `fetch-smoke` route, or provision your own licensed safetensors model offline. |
| Chat template missing | Use a tokenizer with its reviewed template, or explicitly set `prompt_format: "plain"`; no template is silently invented. |
| Context budget exceeded | Shorten practice prompts/menu-compatible config or choose a model with a sufficient context. Prompts are not silently truncated. |
| No supported matrix / byte or parameter budget exceeded | Use a compatible genuinely small model; do not remove all limits. |
| Score is flat / all candidates rejected | Inspect predictions and `reason`; this is a valid result, not an installation failure. Do not loosen the gate just to call it improvement. |
| Interrupted or failed final test | The report records failure; applied accepted patches are undone in `finally`. A hard kill can prevent report finalization, but no source model is saved over. |

For agent permissions see [MCP](MCP.md), for threat assumptions [security](../SECURITY.md), and for tests/fixes [verification](TESTING.md).
