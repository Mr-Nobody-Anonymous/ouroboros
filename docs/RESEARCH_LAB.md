# Research Lab: diagnose, program, compare, submit or withdraw

This is the integrated **0.5 development** research workflow, not a renaming of the [0.4 finite-menu smoke](SELF_EDITING.md). It combines a **frozen same-model editor**, metered target tools, multi-tensor numerical programs, paired controls, sealed reserved validation, independent sessions and a once-only final test.

**Start with the [paper gap checklist](PAPER_PARITY.md).** These tools cover many missing engineering surfaces, not complete paper parity, an OS sandbox or reproduced benchmark gains. A tiny pretrained model actually tried the JSON-action interface and produced **zero valid actions**; the separately scripted integration was rejected. That negative result is recorded, not replaced with a host-written success.

## 1. Install and run the model-driven starter

Use the active branch linked in the [README](../README.md); it is not on the historical default branch/release yet. Python 3.10–3.12, Git and several GB of RAM/disk are recommended for the tested small CPU model.

```bash
python scripts/install.py --profile research
source .venv/bin/activate  # Windows: .venv\Scripts\activate
# New directory only. Includes disjoint public practice/reserved/final fixtures:
ouroboros init ouroboros-research

# Explicit, receipt-bearing data provisioning; not a side effect of running tools:
ouroboros models fetch-smoke --directory ouroboros-research/models/smollm2
ouroboros lab tools
ouroboros lab run --config ouroboros-research/research_lab.json --check-only
ouroboros lab run --config ouroboros-research/research_lab.json
```

The research installer adds `llm,mcp,smoke,math` extras; it does **not** add PEFT training, a model download, datasets, vLLM or an external benchmark harness. For an existing environment use `python -m pip install -e '.[llm,mcp,smoke,math]' -c constraints/ml-tested.txt`. `doctor` lists optional imports without loading them.

The starter is deliberately small/greedy with an exact-match grader. Its Q4_1-derived SmolLM2-135M is **not** the paper's 27B reasoning model and is not reliably able to author JSON weight programs. Invalid actions, withdrawal, budget exhaustion and rejection are valid measured outcomes. The model receives practice feedback only. Its raw replies are archived, not executed as Python.

Check-only validates the JSON schema, split boundaries and limits **without importing Torch or loading a model**. It does not certify enough memory, compatibility, licensing, sandboxing or a correct grader. The loader accepts only local safetensors/tokenizer files, disables remote Python and performs source hashing before/after a configured run.

## 2. Configuration and budgets

`init` creates [`research_lab.json`](../ouroboros/resources/research_lab.json). Paths resolve relative to the config file, not your current shell. Required top-level fields are `schema_version`, `model`, `policy`, `sampling`, `editor`, `grader`, `data`, `output_dir`, `license` and `provenance`; only `damage` is optional. Unknown fields and duplicate/nonfinite JSON keys fail closed.

- **`model`** uses the local model configuration; `max_new_tokens` caps each answer (≤32,768). Context checks reject overlong inputs instead of silently truncating generation prompts. Ordinary FP16/BF16/FP32/FP64 only; quantized mutation is refused.
- **`editor`** is `{"mode":"model","max_new_tokens":512}` for actual generations, or `{"mode":"scripted","actions":"actions.json"}` for a clearly labelled reviewed action list. Never give an action file to model mode. Scripted mode is an integration/host-control path, not autonomous self-editing.
- **`sampling`**: `temperature`, `top_p`, `top_k`, `repetition_penalty`, `presence_penalty`, `thinking` and `thinking_end`. Zero temperature means greedy. Thinking support depends on the model's chat template; setting a flag cannot teach a model to think or guarantee the paper's exact prompt.
- **`grader`** is `exact` (normalized whole final text) or `boxed_math` (explicit Math-Verify 0.9.0 on the last balanced `\boxed{...}` after the thinking close). Cut/unclosed-thinking completions are incorrect. Missing Math-Verify raises an actionable error, never silently downgrades grading.
- **`data`** requires separate `train` (practice), `validation` (reserved) and `test` (final) files in the [Sample format](TRAINING.md). Exact normalized overlap/conflicting targets fail; this is not semantic decontamination. Only practice prompts/references and bounded practice responses enter the editor conversation.
- **`policy`** sets bounds and scientific settings. All omitted fields use the strict [`LabPolicy` defaults](../ouroboros/lab/config.py), not the finite-menu limits. A partial policy overrides only those fields.

Default policy: one session, 24 turns, 20 tool forward requests, 40 numerical-edit actions, three invalid actions and 300 cooperative seconds; ≤300M parameters/1 GiB source bytes, ≤8M elements per edited tensor, eight edited tensors, 250,000 changed entries, 0.1% of all model parameters, 10% relative norm per tensor and 16 MiB sparse payload. Counts/norms use the **actual stored-dtype rounded change**. Whole-tensor rescaling and no-op export are refused. These are maximums, not a recommendation to use all RAM.

`max_forward_calls` counts target **tool requests**, not generated sequences or internal autoregressive token steps. `test`/`control` original-plus-candidate is one request; a sweep charges once per variant. Reads/contrast charge one request even on a subsequent failed read. Initial diagnosis, the one reserved check and final evaluation are separately counted generated sequences but do not consume that tool-request budget. The report separates these quantities. Budgets are cooperative: one in-flight forward may overrun time; use an isolated worker with external limits for hard quotas.

Research resource ceilings can explicitly be raised (up to 8 independent sessions/160 turns/100 tool requests/200 edit actions; see source for every bound). A higher ceiling does **not** add architecture support, GPUs, vLLM or distributed serving. Those paths were not validated for the paper's 27B/hybrid/vision models.

## 3. Available actions

Run `ouroboros lab tools` for required/optional fields and operator formulae. Every action is one JSON object with an `action` name and optional `rationale` (≤2,000 characters). No paths, imports, `eval`, Python source or shell commands are accepted. Notes/directions/saved variants persist between bounded conversation previews.

| Actions | Purpose |
| --- | --- |
| `inventory`, `modules`, `practice` | Page canonical tensor/module names and practice prompts/references |
| `weights` | Target-copy statistics/outlier ranking and bounded numeric prefixes |
| `read` | Actual hidden states, MLP inputs, token log-probabilities or opt-in attention maps |
| `contrast`, `writers` | Measure/store a failing-minus-successful group direction; rank matrix output columns |
| `edit`, `drop` | Atomically stage or remove multi-tensor numerical programs |
| `save`, `restore` | Keep up to four in-memory alternatives |
| `test`, `control`, `sweep` | Paired practice generations, no-op/random/layer controls and dose tests |
| `notes` | Replace bounded persistent notes across history compaction |
| `submit`, `confirm`, `withdraw` | Seal once, reserved check, explicit final decision; no subsequent editing |

Examples of **individual model/tool actions**, not shell commands:

```json
{"action":"read","kind":"hidden","texts":["A long repetitive failed attempt ...","A concise successful attempt ..."],"modules":["model.layers.27"],"weights":"original","pooling":"mean","last":0}
```

```json
{"action":"contrast","module":"model.layers.27","failing":["Failed attempt ..."],"successful":["Successful attempt ..."],"weights":"original"}
```

```json
{"action":"edit","replace":true,"operations":[{"op":"project","name":"model.layers.27.mlp.down_proj.weight","direction":"d0","strength":0.1,"top_columns":1,"max_entries":256},{"op":"project","name":"model.layers.28.mlp.down_proj.weight","direction":"d0","strength":0.1,"top_columns":1,"max_entries":256}]}
```

A measured direction must match each matrix's **output dimension**. Named modules and dimensions are architecture-specific; inventory/measurement come first. Never call a synthetic text contrast a proven causal loop circuit.

The seven numerical operators are: `project`, `clip_outliers`, `set_entries`, `add`, `scale_rows`, `scale_columns`, `low_rank_add`. Projection supports signed strength [-8,8], explicit/top columns and an optional exact largest-entry cap. Flat entries/additions/clipping support dense parameter ranks 1–32; projections/scaling/low-rank factors require matrices. Scalar parameters, buffers, embeddings, final norm and output layers are not editable. Known name/embedding APIs and storage alias protection are not universal architecture introspection; distinct parameters sharing storage fail closed.

Reads are teacher-forced raw text, not sampled continuations. Use `pooling:"positions"`, `last:0`, `stride:4` for bounded positional reads; responses explicitly report token truncation. Full observations are archived; the model sees bounded previews, not an unlimited notebook. Attention maps need `policy.attention_maps:true` and an eager-compatible architecture, with full-allocation bounds. Only tiny random Llama attention is tested.

For practice comparison, use `{"action":"test","ids":[0,1],"weights":"both"}`. Seeds are matched by problem/sample index for baseline, candidate, saved variants and controls. Reports include answer accuracy, cut rate, mean length, finished-answer accuracy, rescues/regressions and problem-clustered paired deltas/SE/sign flips. Multiple samples per problem are averaged **before** uncertainty, never treated as independent problems. The normal interval is approximate; optional `require_significance` is a reserved engineering gate, not a correction for repeated selection.

Controls (`kind`: `no_op`, `random_direction`, `random_delta`, `layer_isolation`) and up to four independently specified sweep variants leave the staged candidate unchanged. Random controls match candidate-support norms **before dtype rounding** and report actual rounding error. They do not prove semantic specificity by themselves.

## 4. Seal, decide, restore

```text
fixed clean editor → diagnose practice → read / edit / paired tests
                                             ↓
                                    submit seals candidate
                                             ↓
                                one private reserved check
                                             ↓
                                  confirm OR withdraw only
                                             ↓
                         host strict gain/retention/statistical gate
                                             ↓
                new independent session starts from original target again
                                             ↓
                   freeze best eligible incumbent before final-test access
                                             ↓
                      once-only paired final test on independent seeds
                                             ↓
                          restore editor / verify source / save report
```

The resident editor is never permanently patched, even after acceptance. In repair mode the editor remains clean but every tool's `original` means the **damaged** target; no clean-reference read tool exists. An explicit operator-only `damage` object gives `parameter_name`, `first_row`, `rows`, `std_multiplier`, `seed`; the location/method is saved separately, not in the editor conversation. Noise replaces consecutive rows using the original matrix's standard deviation, without copying clean values into the proposed repair.

Only aggregate reserved statistics are returned to the editor. After the check, `confirm` cannot override a plateau, forgetting or requested significance failure. Independent sessions do not cumulatively merge edits. The selected incumbent must improve over original and previous incumbent while keeping per-domain floors. Final-test results never feed another session/promotion, and the evaluation plan is hashed before diagnosis. This is an in-process engineering plan, not an external timestamp or adversarial isolation. Repeated validation/session selection still risks overfitting; inspecting final-test results and retuning invalidates their intended role.

## 5. Files and exact replay

Runs use `output_dir/UUID/`. `lab.json` records settings, modes, source/state hashes, per-session status/counters, decisions and final testing. `evaluation_plan.json` fixes data/settings/seeds/analysis. Each `session_XX/` has a diagnosis and `turn_NNN.json` with raw reply/action/full observation plus hashed event references. Reserved/final raw measurements are separate **trusted host artifacts**. They are not safe to share blindly with a proposing agent.

Only an accepted nonempty edit exports `session_XX/bundle/{bundle.json,changes.safetensors}`. The bundle stores exact before/after bytes at sparse indices and binds the **complete before/after state** (unlike the older single-matrix patch format).

```bash
ouroboros lab verify-bundle --bundle /path/to/session_01/bundle
# Explicit extra evaluation, NOT part of the original once-only final test:
ouroboros lab evaluate --config /path/to/research_lab.json --bundle /path/to/bundle --split test
```

Data-only verification needs no Torch and does not reconstruct/validate model behavior. Application needs the exact matching full model state. For a **trusted local Python worker**, `apply_bundle(model, directory)` / `apply_bundle(..., reverse=True)` are explicit mutations with an independent dense-snapshot bound; `use_bundle(model, directory)` is a temporary overlay. Import from `ouroboros.lab.patches`. Multi-target preflight happens before mutation; wrong state, protected aliases, corrupt data or failed output hashes cause refusal/rollback. Hashes establish content identity, not signatures, rights or safety. Never mix base edits into adapter-only SFT and assume its checkpoints own them.

## 6. Import licensed data / plan independent benchmarks

No dataset is downloaded automatically. Convert **separately chosen local files**:

```bash
ouroboros lab import-data --input /data/practice.jsonl --output-dir datasets/practice \
  --format deepscaler --source 'local reviewed collection, pinned revision' --license 'caller-asserted license'
```

Formats: `math` (`problem` plus `answer` or a boxed `solution`), `gsm8k` (`question`, reference `answer` after `####`), `deepscaler` (`problem`, `answer`) and `sample` (existing Sample records). The output is `samples.jsonl` plus `dataset.json` hashes/provenance. Caller-provided labels are **not independently verified**, original splits are not auto-combined, and asserted licensing is not legal certification. Prepare practice/reserved/final separately; no random split can certify semantic decontamination from public benchmarks.

An optional bridge targets the reviewed CLI of the separately installed [lm-evaluation-harness](../external/lm-evaluation-harness):

```bash
# Default: JSON plan only, no model loading/process/install/download and no output directory created.
ouroboros lab benchmark --model-dir ouroboros-research/models/smollm2 \
  --output-dir outputs/gsm8k-independent --task gsm8k --limit 100
# Only after reviewing/installing the upstream tool and provisioning licensed task data:
# add --execute --timeout-seconds 1800
```

It invokes fixed `lm-eval run --model hf` arguments (zero-shot, batch 1, greedy), rejects path/argument injection, uses offline Hub/dataset settings, owns a new output directory and records command/exit/logs. **The actual harness/benchmark has not run here**; only its command boundary is tested with a labelled fake executable. The reviewed revision is recorded, not a cryptographic pin of whatever `lm-eval` is on your PATH. Upstream/task code is trusted; offline flags are not a firewall. This bridge does not install a harness, apply a bundle, export a patched full checkpoint, reproduce the paper sampler or automatically compare gains.

## 7. CLI, MCP and measured evidence

`lab tools/run/verify-bundle/evaluate/import-data/benchmark` are JSON CLI surfaces. The [MCP guide](MCP.md) describes the read-only catalogue/config/bundle/report tools and explicit `--lab-config` allowlist. Execution is a reviewed whole-run offline child, **not** a live unrestricted notebook or arbitrary per-action mutation endpoint.

Reproduce the local plumbing check with an existing receipt-verified model:

```bash
python scripts/run_research_smoke.py --workdir outputs/research-smoke \
  --model-dir ouroboros-research/models/smollm2
```

This runs actual model-authored attempts **and a separately labelled scripted integration**; no fallback actions are passed off as model output. [Local API results](evidence/research-smoke-local.json), [actual CLI results](evidence/cli-research-local.json), [actual MCP results](evidence/mcp-research-local.json), [tests/CI](TESTING.md), [remaining gaps](PAPER_PARITY.md). There is no demonstrated pretrained capability gain, 27B/vision/game study, WeightBench score, persistent arbitrary-Python sandbox, vLLM or tensor-parallel backend.
