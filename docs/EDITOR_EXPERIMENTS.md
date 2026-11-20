# Making the research editor usable—and measuring whether it is

The initial 135M research run produced two prose replies, no usable commands, and no accepted edit. That did **not** isolate model size as the cause. This follow-up addresses identifiable interface, termination, feedback and experiment-design problems without weakening the improvement gate.

## What changed

1. **Optional schema-constrained commands.** `editor.decoding: "json_schema"` filters generated tokens through the closed research action schema, including nested numerical programs. Bounded enums use public target names, practice ids, existing direction/saved ids and paging ranges when the domain has at most 4,096 values. The model still generates actions, tensor names, indices, strengths and other values; there is no host-written edit fallback. The optional MIT `lm-format-enforcer==0.11.3` integration is separate from answer generation.
2. **Small prompts and persistent feedback.** The editor gets compact tool signatures, graded practice examples and explicit staged-tensor/direction/saved-variant state, not the complete machine catalogue every turn. Context compaction keeps the latest observation/error or stops explicitly; it never silently removes that error. Long text previews retain both beginning and end; full answers remain archived. Dynamic wall-clock values stay in host reports, **not editor context**, so paired conditions can have byte-identical initial prompts.
3. **Actionable, metered repairs.** Parse failures retain the actual JSON error. Semantic errors name missing/unexpected fields. Feedback supplies required/optional fields and the closed language, not a generic “unknown action.” Every failed reply still consumes a turn and invalid-action budget. There are no unlimited hidden retries.
4. **Correct stop-token handling.** The old converted SmolLM2 tokenizer declared token **1**, a start-of-turn token, as EOS; the model and generation config declared token **2**, end-of-turn. In-memory loading now respects model-declared EOS, aligns the tokenizer/padding and makes the format enforcer agree. The provisioner fixes this metadata before saving **new** conversions. Existing receipt-bearing source files are never overwritten. Only the terminal stop token is removed; thinking delimiters remain visible, and cuts/open thinking still fail grading.
5. **Practice-only readiness and real contrasts.** `lab diagnose` measures practice without editor replies, edits, reserved generation or final generation. It reports all-wrong/all-correct/mixed outcomes, strict grades, completion lengths and cut/open-thinking warnings. `contrast_practice` generates and grades requested practice ids, then measures actual failed-versus-correct completion cohorts. If either cohort is missing or a complete contrast input cannot fit, it returns `available: false`; it never substitutes reference answers or synthetic positive texts. Both contrast tools preflight **all complete inputs before any hidden forward**, refusing an overlong input instead of measuring a prefix that could omit the completion.
6. **Bounded exploration and lower memory overhead.** The starter has separate editor/answer sampling, 128 answer tokens, 384 command tokens, 16 turns, four invalid replies, shorter history and an explicit no-progress limit. Three identical consecutive read-only commands stop as **`stalled`**, not a model withdrawal or success. Integrity hashes now stream tensor buffers instead of allocating/concatenating full byte copies, preserving the existing hash/patch format. Reports flush after each archived turn; SIGKILL/OOM can still prevent final restoration verification.

A grammar fixes **syntax**, not scientific reasoning. Numerical ranges are also schema annotations; the decoder does not enforce every bound. Host name/shape/protection/finite/rounded-entry/norm/byte checks always run. The first structural-only comparison exposed a real limitation: syntactically valid `offset: -1` passed the library grammar and was rejected by the host. Schema version 2 adds explicit bounded enums for paging/ids rather than trusting ignored numeric range annotations. Very large domains still fall back to host-checked structural fields, not enormous enum allocations. After sealing, only `confirm` and `withdraw` are grammatical **and** allowed by the host.

## Measured outcome: commands work, useful editing still does not

The latest completed **paired comparison 04** used the same receipt-verified, Q4_1-derived **134,515,008-parameter SmolLM2**, answer/editor settings, seeds, practice baseline and host policy. Both conditions had the same initial prompt SHA-256, `a3315b1d4bd7cdb628bb5202d16f90c28887f7e9ab658aebbb943c482b611bcd`.

| Observation | Unconstrained | State-bound JSON schema |
| --- | --- | --- |
| Model editor generations | 4 | 3 |
| Parsed JSON replies | 0 | 3 |
| Host-valid commands | 0 | 3, all `inventory` |
| Invalid replies/actions | 4 | 0 |
| Staged edits / changed entries | 0 / 0 | 0 / 0 |
| Practice comparison actions / accepted patches | 0 / 0 | 0 / 0 |
| Stop | Invalid-action budget | Host `stalled` after repeated inventory |
| Full restoration / source files unchanged | True / true | True / true |

Both measured **1/18 correct practice answers**, with **2/18 cuts**, under unchanged strict grading. Syntax/host validity improved; neither condition authored an experiment, edited weights, or demonstrated a capability gain. This is one fixed order/seed/checkpoint on repeated public diagnostics, not an independent generalization study or proof that a larger model is necessary/sufficient.

The earlier **structural-only comparison 02** is preserved: unconstrained 0/4 parsed/valid; schema 5/5 parsed but only one valid inventory and four negative-offset rejections. Its original plan accidentally included dynamic elapsed time in prompts; the initial prompts were **not identical**, so its counts are not a clean decoding-only causal comparison. The **schema-only follow-up 03** tested bounded enums: three valid inventory calls, then `stalled`, no edits. It is not another paired comparison. The reviewed [comparison evidence](evidence/research-interface-local.json) keeps all three completed runs, plans/source hashes and this correction separately.

Additional real checks, kept distinct from autonomous editing:

- **CLI `lab diagnose`:** 18 practice generations, 1 correct / 17 incorrect; zero editor replies, edits, reserved or final generations. Restoration/source checks passed. [CLI evidence](evidence/research-diagnose-cli-local.json).
- **Official-SDK stdio MCP:** initialized, discovered 12 tools, checked config/catalogue, ran actual model/schema commands and read back the completed report. Three valid inventory calls, no edits/acceptance, restoration/source checks passed. [MCP evidence](evidence/mcp-structured-editor-local.json).
- **Explicitly host-driven contrast verification:** 18 actual pretrained practice answers; one correct and four failed completions selected for a 576-dimensional unit direction, **zero truncated read inputs**. Re-reading those **same cohorts** with an intentionally inadequate two-token limit refused before any hidden forward or direction creation. Zero edits or held-out inference. This checks the tool and refusal, not autonomous discovery, a causal circuit or a beneficial edit. [Contrast evidence](evidence/research-graded-contrast-local.json).

The first development comparison was **OOM-killed and incomplete**. Its interrupted condition has **no successful restoration claim**; it is not counted as a pass. Subsequent completed runs were serial with model-heavy tests. Streaming hashes and shorter history are not RAM quotas; use external memory/time isolation for hard limits. Historical [initial smoke evidence](evidence/research-smoke-local.json) is unchanged. Each plan identifies its tested source snapshot; the later full-text contrast fix was verified directly and is not retroactively attributed to the inventory-only runs.

## Run the improved starter

```bash
python scripts/install.py --profile research
source .venv/bin/activate
ouroboros init new-research-project
# Explicit model provisioning, or point the config at an existing licensed local model:
ouroboros models fetch-smoke --directory new-research-project/models/smollm2
ouroboros lab run --config new-research-project/research_lab.json --check-only
# Practice only: inspect suitability before spending an editing budget.
ouroboros lab diagnose --config new-research-project/research_lab.json
# Actual model-generated commands; no scripted fallback:
ouroboros lab run --config new-research-project/research_lab.json
```

The scaffold preserves the old self-edit fixtures and adds **18 practice / 18 reserved / 18 final** owned arithmetic/copying diagnostic examples. They are not MATH/GSM8K, independently decontaminated tasks or a general-capability benchmark. The copying family was selected using **practice-only** suitability pilots, not reserved/final outcomes. Targets are independently specified arithmetic results/words; they are never rewritten to match model predictions. Do not directly compare rates on the old and new task suites, or call prompt/task calibration a weight-edit gain.

For a suitable local model, import independently chosen licensed data and fix the grader, prompts, budgets and split policy **before** evaluation. A mixed practice baseline is useful feedback, not proof that a successful intervention exists. All-wrong diagnostics and remaining cuts should prompt investigation, not lenient grading, mid-run budget changes or manufactured successes. The 128-token starter reduces the old premature limit; it does not guarantee every answer finishes.

### Configuration

```json
"editor": {
  "mode": "model",
  "decoding": "json_schema",
  "max_new_tokens": 384,
  "sampling": {"temperature": 0.3, "top_p": 0.95, "top_k": 20, "thinking": false}
}
```

`sampling` at the top level still governs **answers**, including all baseline/candidate/validation/final pairs. `editor.sampling` is independent and defaults to greedy/no thinking when omitted. JSON commands cannot have a thinking preamble. `decoding` defaults to `unconstrained` for existing API/config users; the new scaffold opts in explicitly. Missing structured dependencies raise an installation error, **never a silent fallback**. Check-only and `lab tools` need no Torch or decoder package.

The generated-command grammar is a practical bounded subset (for example, at most 16 operations and 240 rationale characters). The trusted Python/scripted API retains its documented larger host limits. Neither interface permits arbitrary Python, paths or model loading.

### A real-output contrast

```json
{"action":"contrast_practice","module":"model.layers.15","ids":[0,17],"weights":"original","last":16}
```

Choose actual module names and practice ids for your model/data; the example does not promise both cohorts exist for those ids. This charges one generation request, plus one attempted hidden-read request **only when both cohorts exist**. Texts are the original practice instruction, a newline, and the actual completion, not a recreated chat transcript. Stored sources retain problem/sample/seed/verdict metadata. `practice_records` stay separate from the hidden `measurement` so neither overwrites the other.

Each full text must fit **both** `policy.max_read_tokens` and model context. No tail/prefix cropping or reference substitution makes a contrast appear available. When too long, `contrast_practice` retains the measured grades/completions and returns a bounded refusal with token limits; direct supplied `contrast` raises a metered validation error. Set appropriate fixed budgets before another experiment, never in response to held-out results. Ordinary `read` retains explicitly reported prefix truncation. `last` pools the requested final positions **after the whole input is read** (default for practice contrasts: up to 16 positions).

Different questions/content can confound the direction; it is not a discovered causal circuit. Ordinary `contrast` remains available for explicitly supplied texts, including clearly labelled synthetic integration fixtures. Only actual graded model outputs may be called successful practice completions.

## Reproducible interface comparison

```bash
python scripts/run_research_experiment.py \
  --workdir outputs/research-interface \
  --model-dir new-research-project/models/smollm2 \
  --compare-unconstrained --turns 16 --max-seconds 600
```

The workdir must be new. The script requires a verified existing model receipt; it never downloads. It writes condition order, data/config/implementation hashes, schema version, sampling/budgets and selection limitations **before** either run. The intended setting difference is `editor.decoding`; both start from the same full model state and practice baseline. Dynamic timing is excluded from editor prompts, initial prompt hashes must match, and a mid-run implementation/resource change is an error. Later histories naturally differ as generated actions/observations differ. One condition's outputs never feed the other editor; final results do not select a condition or another candidate.

`experiment_summary.json` separates parseable JSON, valid host actions, staged edits and practice comparisons—valid inventory/notes are not self-improvement. Source-bound plans are engineering records inside a trusted host, not externally timestamped scientific preregistrations. See [testing/CI](TESTING.md) for current local versus remote verification.

## What this does not fix

A valid JSON command is not a useful hypothesis. Repeated inventory, unsupported edits, lack of successful practice outputs and failure to beat controls remain possible. The strict original/incumbent improvement, per-domain retention, optional significance, sealed reserved check and once-only final-test ordering are unchanged. No forced acceptance, automatic target relabelling, relaxed exact matching, hidden edit scripts, demonstrated pretrained gain or full paper parity is implied.

The optional decoder source was actually cloned and inspected at [817f944fcc8851917c40300a47f62d1defc3ffc3](https://github.com/noamgat/lm-format-enforcer/tree/817f944fcc8851917c40300a47f62d1defc3ffc3): schema parser, parser config, token-enforcer cache and Transformers prefix integration. Ouroboros calls the separately installed pinned package; no upstream library code is vendored. [Source/integration receipt](evidence/structured-decoder-source.json), [attribution](../THIRD_PARTY_NOTICES.md), [research tools](RESEARCH_LAB.md), [remaining paper gaps](PAPER_PARITY.md), [security boundary](../SECURITY.md).
