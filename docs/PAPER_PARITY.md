# Paper gap audit: what was missing, what was added, what is still missing

**Scope:** the self-rewriting paper [Language Models Can Rewrite Their Own Weights, v2](https://arxiv.org/html/2611.21885v2), particularly §2, Algorithms 1–4, Appendix D and §E.9, plus the earlier [MIT CSAIL weight-space programming blog](https://rikiyat.github.io/blog/weight-space-programming/), experiments 1–8 and its limitations/FAQ. These are different works: the former uses the same model as editor/target; the latter uses a stronger external editor. This is a source-linked engineering audit, **not a claim of complete paper parity or reproduced scientific results**.

The comparison baseline is Ouroboros **0.4**: a real same-model **finite menu** over one matrix per trial, tiny greedy/exact-match fixtures, reversible accepted chains, CLI and MCP. The additions are the **0.5 development research workflow** under [`ouroboros/lab/`](../ouroboros/lab). The original smoke/SFT paths remain available; they are not silently changed into this protocol.

## New discovery work is a distinct methodological profile

The [closed-loop discovery follow-up](DISCOVERY_EXPERIMENTS.md) adds prospective hypotheses,
complete attempted-intervention ledgers, actual feedback/revision, explicit sensitivity,
host/random/no-edit/model controls, eleven arms/two seeds, a broader 240-example five-role
set and post-selection subsequent-edit retention. Development seals **before its first
look**, then reserved independently confirms. The strict 44-submission family correction
is an additional engineering protocol, not the paper's exact original method.

These additions close measurable experimental gaps but **do not** turn tests or gradient-
assisted host controls into no-gradient paper parity. They also do not certify independent
semantic circuit discovery, learned future self-improvement or a beneficial pretrained edit.
The initial 135M local study is running and larger 360M Instruct study configured at this
implementation checkpoint; their results must be read separately from the earlier negative
runs and the 412 passing tests. The general 19-action lab comparison below is preserved;
the opt-in 25-action discovery profile additionally requires explicit per-edit hypotheses.

## 1. Self-rewriting: missing/partial features and disposition

**Implemented** below means owned executable tooling with tests. It does not mean a model successfully used it to improve a benchmark. **Partial/different** and **missing** are intentional, visible limits, not hidden TODOs behind a success flag.

| ID | Source feature | Gap in 0.4 | Current disposition |
| --- | --- | --- | --- |
| P01 | Fixed editor weights; candidate applies only to requested target runs (§2.2, Alg. 1) | Accepted rounds also changed the next proposer | **Implemented in `lab`**: one resident frozen editor; all target overlays restored before each reply; full editor hash checks. No second full model copy needed. Old `self-edit` still has its documented chain semantics. |
| P02 | Starting diagnosis: graded practice answers, accuracy, token-limit/repetition and finished-answer accuracy (§2.2) | Only a few exact-match practice predictions | **Implemented**: actual records plus cut share, zlib-tail heuristic, finished accuracy, lengths, strict verdicts and practice-readiness warnings. `lab diagnose` measures practice only; no held-out generation or edits. Practice only enters conversation. |
| P03 | Model authors a rationale and numerical weight programs, rather than choosing a small fixed edit menu | No free-form action/program generation | **Implemented, narrower language**: model-generated JSON actions and composable numerical operations, not label choice. Optional JSON-schema token filtering and compact prompts improve command syntax; parse/field errors reach metered repair feedback. Invalid replies are recorded/rejected; no scripted fallback. Reliable authorship is a model capability, not guaranteed by this API. |
| P04 | Persistent general Python CPU kernel over weights, NumPy and session files (Alg. 4) | Not present | **Missing intentionally**: no arbitrary Python executor. Persistent directions, notes, current edit and four saved variants exist in the closed data-only language. This is not an equivalent Python kernel or OS sandbox. |
| P05 | Load/write/drop/list target tensors, arbitrary same-shape replacements | Only one directional matrix ablation | **Partial**: inventory/statistics/prefix reads, atomic multi-tensor programs, save/restore/drop; finite dense parameters of rank 1–32. Flat entries/additions/clipping support higher ranks; low-rank updates, row/column scaling and projections require matrices. No scalar-parameter/buffer edits, imports or file-based replacement. Scalar edits would necessarily be whole-tensor rescaling under this policy. |
| P06 | Exclude embeddings, unembedding, final norm and whole-tensor rescaling (§2.3) | MLP menu avoided most by construction, generic trial API did not enforce this paper policy | **Implemented in `lab`**: protections include tied/storage aliases for supported layouts; distinct shared-storage parameters refused. Known architectural names supplement embedding APIs, not universal architecture introspection. |
| P07 | Hidden states after chosen layers on model-chosen texts | Vector reduction utility existed, no instrumented read | **Implemented**: named-module hooks, per-position/mean pooling, last positions or stride, explicit read truncation reports, element/logit limits and hook cleanup. Raw teacher-forced text, not a claim of semantic diagnosis. |
| P08 | MLP activations and token log-probabilities (Alg. 2, Appendix D) | Missing | **Implemented**: selected MLP-input hooks and actual next-token log-probabilities. |
| P09 | Practice generation: original/edited/both, same problems and seeds (§2.2, Alg. 3) | Greedy evaluation; one global seed reset | **Implemented**: seeds derived from normalized problem identity, sample index and a fixed seed base; original/edited/saved variants use identical pairs; multiple samples per problem. Same-seed cross-device equivalence is not guaranteed. |
| P10 | Controlled sampling, thinking closure, final boxed answer grading, long token budgets | Greedy, ≤64 tokens; strict text equality only | **Implemented tooling**: temperature/top-p/top-k, presence and repetition penalties, thinking-template argument/closing marker, answer cap up to 32,768; explicit optional Math-Verify 0.9.0 final-box grader. Cut/open-thinking answers fail closed. This is not automatically the paper's exact prompt/model/sampler. |
| P11 | Paired mean changes, per-problem SE, wrong→right/right→wrong and cut changes (Eq. 3) | Missing uncertainty and paired outcomes | **Implemented**: per-problem clustering, paired accuracy/cut deltas, SE, approximate intervals, rescues/regressions and original-cut/finished cohorts. Repeated answers are not counted as independent problems. |
| P12 | One-sided sign / problem sign-flip tests (Appendix D.3) | No statistical reporting | **Implemented**: exact sign flips for ≤16 nonzero problem deltas, otherwise seeded Monte Carlo with plus-one correction; optional reserved-check significance gate. Approximate normal intervals and repeated validation are not a sequential/multiple-testing guarantee. |
| P13 | Forward/code/turn/time budgets; bounded output and conversation restart | Only ≤4 smoke rounds | **Partial/different**: metered target requests, actual generated-sequence counts, numerical-edit actions, invalid actions, turns and cooperative time; compact context plus persistent notes. No Python code-run counter, 90k-token persistent notebook, hard OS quotas or automatic crash resume. Failed expensive tool attempts are charged, unlike the paper's free failed requests. |
| P14 | Compare several saved edits side by side (§E.9) | Sequential menu only | **Implemented**: up to four saved variants, common seeds and paired comparisons; explicit total-answer caps. |
| P15 | Reserved practice check once; then SUBMIT/WITHDRAW with no more editing (§E.9) | Validation repeatedly used during all trials | **Implemented**: practice tools cannot request validation/test; `submit` seals, runs one reserved check, then only `confirm`/`withdraw`. Host still rejects plateaus/regressions. This stricter host gate is a protocol difference. |
| P16 | Eight independent sessions restart from base (§E.9) | Accepted chain across rounds | **Implemented**: configurable 1–8 independent sessions, fixed editor and same target start. Keep the best eligible validation incumbent without merging sessions. Unlike the paper's table of every submitted edit, final evaluation here is of the selected incumbent only. |
| P17 | Separate held-out/new-seed evaluation fixed before inspection (Appendix D.3) | Tiny post-selection final split, no analysis plan | **Implemented tooling**: hashed evaluation plan written before diagnosis; fixed grader/settings/splits/sample counts/seeds/test; winner frozen before once-only final evaluation. Not externally timestamped, not semantic decontamination, no replication of the paper's two-look design. |
| P18 | Damage consecutive rows with scaled Gaussian noise (§2.4, Appendix D.2) | Missing | **Implemented fixture capability**: explicit operator-only row/noise specification; editor remains clean, every original/read tool sees the damaged target; damage location stays outside conversation. No autonomous pretrained repair result established. |
| P19 | Find outlier matrices and repair from damaged values, not a clean reference (§E.3) | Missing weight statistics/repair operations | **Implemented tools**: median/max/outlier and row-norm statistics; median-relative clipping; explicit entries. These are aids, not a built-in claim that the model diagnosed the fault correctly. |
| P20 | Multi-layer signed projections and 1D time-step/log-decay edits (§E.4, §E.9) | One matrix, coordinate-only, strength ≤1 | **Implemented numerical operators**: signed projection strength in [-8,8], multiple matrices, selected/top writing columns, sparse entry cap, finite 1D additions/entries. Actual rounded global count/fraction, per-tensor norm and total byte gates. Model-specific interpretation of a bias remains the editor's responsibility. |
| P21 | Same-model 2B/4B/9B/27B scale ladder; hybrid Qwen3.5 | 135M smoke only | **Not run / model-specific integration unverified**: configurable research resource ceilings can exceed smoke limits, but only tiny Llama and SmolLM2 are verified. No 27B checkpoint, CUDA device or scale-ladder experiment here. Raising a budget does not add architecture support or memory. |
| P22 | vLLM, high-throughput long generation, same engine reset for editor (Appendix D) | Single-process Transformers greedy path | **Missing**: research backend remains serial, per-prompt local Transformers on one configured device. No vLLM, tensor-parallel/multi-GPU/distributed serving or paper-throughput claim. |
| P23 | Repair transfer to six image benchmarks (§E.6) | No vision-model evaluator | **Missing**: no vision loader/processor, image benchmark adapters or transfer measurements. |
| P24 | Actual benchmark gains and repair recovery | No gains on tiny public smoke | **Not established**: tools/tests are not scientific evidence. No full MATH-500/GSM8K/IFEval/GPQA/vision benchmark, safety/locality study or paper improvement reproduced. The new discovery profile separately tests a fixed 60-problem GSM8K selection, not the standard full benchmark. |

## 2. Additional Astra/blog capabilities

| Source experiment | Addition or remaining gap |
| --- | --- |
| 8: productive/repetitive contrast, writer columns, signed/dose tests | **Implemented tools**: actual hidden/MLP reads, stored contrast directions, column ranking, multi-layer/column projection and independent strength sweeps. `contrast_practice` derives real failing/correct cohorts from graded practice answers, or reports unavailable; supplied synthetic contrasts remain explicitly different. Neither proves a causal repetition circuit. |
| 8: no-op, norm-matched random and layer-isolation controls | **Implemented**: no-op; random direction or random delta on candidate support with per-tensor matched pre-rounding norms; actual rounding error reported; isolate selected edited tensors; candidate remains unchanged. Different support/dose choices still need scientific justification. |
| 7: attention-only inspection | **Implemented for one compatible model** with explicit eager-attention opt-in/allocation limits; real random-Llama maps tested. **Missing**: the complete paired-model sycophancy dataset/training/identification study. Attention maps alone do not prove a sycophancy detector. |
| 1–3: identify games, diffusion shapes, corpus/induction heads from raw weights | **Not implemented as autonomous tasks**. Numeric inspection exists, not a semantic-reading oracle or the original no-forward experimental harness. |
| 4: Kinetix car-policy repair | **Not reproduced/integrated**: no Kinetix environment, recorded scene/rollout scorer or paper policy experiment. Generic bounded tensor arithmetic is not the experiment. |
| 5–6: write MinAtar/Go networks from zero with no training | **Missing**: no game environments, network compiler, search-free rollout tools or zero-initialized network-synthesis workflow. The relative-norm edit policy deliberately does not treat writing a zero network as an ordinary patch. |
| Stronger external Astra editor | Trusted Python clients can drive `LabSession.act`; default automated mode uses the target model itself. **No Astra/provider client, paid calls or frontier-model capability is bundled.** MCP can run a reviewed whole workflow, not provide an unrestricted live notebook. |
| Real benchmark preparation/evaluation | **Partial infrastructure**: explicit local MATH/GSM8K/DeepScaleR import and offline, opt-in installed `lm-eval` command driver. No implicit Hub data download or standard-task result. WeightBench stays a separately set-up reference harness. |

## 3. Local measurements, not expected outcomes

The **initial** 0.5 run used the same pinned Q4_1-derived **134,515,008-parameter** SmolLM2 as 0.4. Its historical results remain below; [the follow-up](EDITOR_EXPERIMENTS.md) has separate constrained/unconstrained comparison evidence and unchanged gates:

- **Model-authored mode:** two genuine model generations, **zero valid JSON tool actions**, two invalid actions, stopped at budget. No host-script substitution. The small model did **not** demonstrate a useful autonomous research policy.
- **Scripted integration mode:** all **16** actions succeeded; **10** target forward requests, actual hidden/MLP/logprob reads, a measured synthetic contrast, writer ranking, two-matrix edit, paired/saved tests, norm-matched/no-op/layer controls and a dose sweep. **1,152 entries** changed across two matrices. Reserved exact-match accuracy **0.0→0.0**, **rejected**, no accepted bundle. Largest control norm rounding discrepancy: **1.8843233596132713e-08**.
- Both modes restored the full state and left source files unchanged. State SHA-256: `6af734f2a618851e913950745c62fca65a4664cf6a9474b1b03577b1cc5329c8`.
- A separate **actual CLI** run of `examples/research_scripted.json` completed the same 16 actions, 1,152-entry edit and plateau rejection/restoration. A separate **actual official-SDK stdio MCP** child ran model-authored mode and report readback: same two invalid replies, no accepted edit, restoration/source checks passed.

[Initial local evidence](evidence/research-smoke-local.json) · [initial real MCP evidence](evidence/mcp-research-local.json) · [test/remote CI status](TESTING.md). The initial research implementation also passed all six jobs in both push/PR GitHub runs, with the same measured failure/rejection outcomes ([API-verified CI evidence](evidence/research-smoke-github.json)); those historical runs do not verify later code.

The separate [editor follow-up](EDITOR_EXPERIMENTS.md#measured-outcome-commands-work-useful-editing-still-does-not) completed a matched-initial-prompt comparison: **0/4 valid unconstrained commands versus 3/3 valid schema inventory calls**, then host `stalled`; zero edits/practice comparisons/accepted patches/gains. A real graded-output direction and overlong-input refusal were verified **by the host**, not autonomously discovered. These remain public diagnostic fixtures on a quantization-derived model, **not** the paper's models, data, gains or causal-discovery experiment.

## 4. What was actually cloned and adapted

The three relevant submodules were initialized at their existing locked commits, inspected, not merely linked:

- **Original experiment logs** `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`: inspected `LICENSE`, experiment 8 README, `apply_edit.py`, `evaluate.py`, `mathqa.py`, and harness layout. Adapted its projection/column intervention, exact base→edit reconstruction, final-box grading and separate-evaluation contracts into owned bounded tools; did **not** run its hidden-path/vLLM scripts or copy benchmark/log data. Scripts are MIT; logs/networks are CC BY; MinAtar excerpts have a separate GPL license. See [attribution](../THIRD_PARTY_NOTICES.md).
- **EleutherAI harness** `d6de81643928d653435c431bae19945d41d32520`: inspected `lm_eval/_cli/run.py` for actual `lm-eval run` arguments. The owned driver targets a separately installed executable with explicit offline settings; installed harness/real benchmark execution remains unverified.
- **WeightBench** `2acd60172fc9758eb9030b1cab4a2ff29be5c993`: inspected README, package entry points and requirements. Its OS harness/42 tasks were **not** silently imported or executed. Source checkout ≠ benchmark result.

No official self-rewriting code repository was verified by this audit. Its protocol additions are owned implementations from the public methods/appendices, not a claim to have cloned unavailable author code. Four other submodules may remain uninitialized; the locked inventory continues to distinguish references, installed integrations and absent trees.

## 5. Use the integrated workflow

Follow [Research Lab](RESEARCH_LAB.md). Run `ouroboros lab tools` for the machine-readable action/operation catalogue. To close the remaining scientific gaps, provision licensed paper models/data and sufficient hardware, use a separately isolated grader, preregister the real comparisons and run them. **Adding APIs cannot establish those results, and a missing OS sandbox cannot safely be replaced with `exec`.**
