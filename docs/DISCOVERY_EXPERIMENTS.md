# Closed-loop discovery: mechanisms before claims

Ouroboros now has a **separate, hypothesis-driven discovery profile**. The earlier
[negative editor runs](EDITOR_EXPERIMENTS.md) are preserved, not relabelled as successful
self-improvement. Valid JSON, a nonempty edit, a useful edit, generalization, retention and
learning how to improve are **different measurements**.

## Status and what counts

At the implementation checkpoint, **412 local tests passed**, including a real neural
token-classifier positive control, genuine Transformers autograd/finite-difference checks,
sealed development/reserved checks, exact numerical replay/reversal and subsequent-edit
retention. The classifier is deliberately hand-designed, **not a pretrained LLM**; its
50%→100% accuracy is a mechanism regression test, not an independent capability result.
Its reworded feature-identical examples do not establish a research significance claim.
The no-Torch core+MCP suite also passed **282** tests; an installed non-editable wheel's
five-role scaffold/catalogue/check-only and all **16 JSON/TXT resources** were checked
outside the source checkout. [Implementation/distribution receipt](evidence/discovery-implementation-local.json).

The first source-bound **pretrained 135M study is running locally**. The larger, original
**360M Instruct checkpoint study is configured for GitHub**, not yet verified at this
implementation checkpoint. Do not infer their outcomes from the test suite. Results and
failed trials will be recorded separately when those runs finish. No beneficial autonomous
edit, generalization gain, retention of such a gain or learned self-improvement is currently
claimed.

| Level | Required evidence |
| --- | --- |
| Command validity | Parsed command passes host interface checks. An inventory query is not an edit. |
| Edit validity | Actual stored-dtype nonzero parameter delta, with unchanged protection/count/fraction/norm/byte rules. |
| Edit usefulness | Strict development **and** independent reserved macro-accuracy gain, no category regression, configured problem-clustered significance, explicit confirmation. Lower NLL alone cannot pass. |
| Generalization | Selected and terminal diagnostic candidates frozen before final evaluation; strict gain and uncertainty reported without reselection. A rejected candidate's final result cannot rescue it. |
| Retention | Predeclared subsequent edit, separate retention questions, category scores at least both original and pre-followup candidate. Equality is retention, not improvement. |
| Future self-improvement | A separately trained editing-policy checkpoint and unseen editing tasks would be needed. **Not implemented/demonstrated by this experiment.** |

The primary autonomous milestone requires a receipt-verified LLM's own numerical edit,
nonempty valid changes, both selection gates, final gain, subsequent-edit retention and
restoration. **No host-written fallback can qualify.** Host controls remain useful but
explicitly non-autonomous evidence.

## Run it

Use the [research installer](SELF_EDITING.md) and explicitly provision a licensed model.
Nothing in inference or `--check-only` downloads a model.

```bash
python scripts/install.py --profile research --dev
source .venv/bin/activate
ouroboros init discovery-lab --discovery
ouroboros models fetch-smoke --directory discovery-lab/models/smollm2
ouroboros lab tools --discovery
ouroboros lab run --config discovery-lab/discovery_lab.json --check-only

# All registered controls/ablations, two fixed seeds; can take substantial CPU time.
# Use a NEW output directory for every experiment; never overwrite an earlier plan.
ouroboros lab experiment \
  --model-dir discovery-lab/models/smollm2 \
  --data-dir discovery-lab/discovery \
  --workdir outputs/discovery-135m-01 --profile 135m
```

For a single configurable model-editor session, use
`ouroboros lab run --config discovery-lab/discovery_lab.json`. This is **not** the full
matched study. Explicitly approved MCP `run_research_lab` configs can use the same profile;
`research_tools(discovery=true)` exposes its catalogue. Approval now binds **all five role
files** as well as the config and any action file. There is no arbitrary shell/download tool.

A larger-model machine with Hub access can explicitly run:

```bash
ouroboros models fetch --repo HuggingFaceTB/SmolLM2-360M-Instruct \
  --directory models/smollm2-360m --max-bytes 2147483648
ouroboros lab experiment --model-dir models/smollm2-360m \
  --workdir outputs/discovery-360m-01 --profile 360m
```

`models fetch` resolves and receipts an immutable upstream revision **before** the study
plan or inference. Replay should use that recorded revision, not assume `main` stayed fixed.
The explicit 360M profile raises only the parameter/on-disk resource cap (400M / 2 GiB),
**not** permitted edit fractions, norm limits, grading or promotion thresholds. The 135M
Q4_1-derived and original 360M checkpoints differ in precision/source as well as size; that
comparison alone cannot identify a pure size effect.

## Fixed data and roles

The [packaged manifest](../ouroboros/resources/discovery/manifest.json) binds **240 selected
examples**, with four balanced categories: arithmetic, copying, compositional reasoning and
GSM8K word problems.

| Role | Examples | Editor access | Use |
| --- | ---: | --- | --- |
| Practice | 32 | Prompts/references/actual responses and tool feedback | Diagnosis, hypotheses and adaptive search |
| Development | 32 | Aggregate only after irreversible sealing | First strict accuracy/significance gate |
| Reserved | 48 | Aggregate only after development passes, still sealed | Independent confirmation, never adaptive search |
| Final | 64 | No editor observations | Frozen selected/terminal-diagnostic comparisons only |
| Retention | 64 | No editor observations | Four-cell fixed subsequent-intervention probe |

**60 problems are independently authored GSM8K examples**; the remaining 180 are owned
synthetic diagnostics, not an external benchmark. [OpenAI's repository](https://github.com/openai/grade-school-math)
was actually cloned at `3101c7d5072418e28b9008a6636bde82a006892c`; train/test/license
bytes and selected raw-row hashes are recorded. Its code is **not executed**. Its
[MIT notice](../ouroboros/resources/discovery/GSM8K_LICENSE.txt) is included in source,
wheel and optional scaffold. Practice/development/reserved draw disjoint upstream train
rows; final/retention draw disjoint upstream test rows. Rationales are not editor inputs.

Selection uses a fixed SHA256 ordering and a declared question-character/numeric-target
filter, **not model answers or performance**. Commas are removed from GSM8K numeric reference
strings by the documented adapter. Final/retention synthetic prompts shift templates,
operand ranges or copied content; they do not silently change grading. Exact prompt
separation is checked across all five roles; **semantic/pretraining decontamination is not
claimed**, and template-related examples can remain correlated.

Rebuild into a new directory, explicitly allowing a clone only when requested:

```bash
python scripts/prepare_discovery_data.py --fetch --output-dir outputs/rebuilt-discovery-data
```

## Controls and competing diagnoses

Each checkpoint uses independent sessions at editor/control seeds **17 and 42**, with the
same target starting hash, answer budget, scoring and strict gates. Answer generation is
greedy: these seeds vary editor sampling, random controls and statistics, not independent
stochastic replicas of greedy answers. No run restarts from another arm's selected weights.

| Arm | Intervention / question |
| --- | --- |
| `no_edit` | Ends without a proposed edit; still measures baseline/final and the fixed follow-up probe. |
| `human_directed` | **Host-authored**, explicitly gradient-assisted positive control: final-block post-attention norm, top 256 practice-gradient entries, two predefined relative doses (0.01, 0.03), rollback harmful trials. Same hypothesis/edit tools and gates as model arms. |
| `random_valid` | **Host-authored** random direction on the corresponding host trial's tensor/support, matched to its actual delta norm. Seeded independently; first trial is rolled back. Support is gradient-informed, not a fully gradient-free null. Missing reference means unavailable, not an unmatched fallback. |
| `unconstrained` | Model emits its own commands without token grammar; no numerical fallback. |
| `schema` | Same initial prompt/seed/sampling, model-authored actions under structural grammar. Initial prompt hashes must match the unconstrained arm. |
| `model_directed_gradient` | Same free workflow, with opt-in practice sensitivity access. The model still generates its own target/operation/indices/values. |
| `schema_no_feedback` | Actual edited-practice outcomes are archived but withheld from subsequent editor prompts; edited-weight practice queries are host-blocked. Original reads remain available. |
| `schema_full_context` | Expanded tool/operation explanations versus the compact prompt. This changes explanation/context load, not a pure model-capacity intervention. |
| `schema_sparse_ops` | Only explicit entry/add operators, with the same protection and acceptance rules. Tests whether a larger operation language helps or confuses. |
| `schema_staged` | Host narrows workflow phases to diagnosis/hypothesis→edit→feedback/revision/submit. No host-written parameter choice or numerical edit. Tests action-planning scaffolding separately. Stopping is always allowed. |
| `schema_staged_gradient` | Same staged workflow with sensitivity access; separates this access from workflow scaffolding. |

The host positive control is run **before** its matched random and model-editor arms. If it
only decreases loss, it establishes **local loss controllability**, not that the full useful
accuracy-edit mechanism has been positively controlled. If the model never requests gradients
or edits in an access ablation, that cannot distinguish failed reasoning from unused tools.
A few trajectories cannot establish a causal diagnosis of error-to-parameter mapping,
parameter selection, translation to numbers, context limits or insufficient training/size.
The result includes an explicit evidence/limitations table rather than asserting one cause.

## Hypothesis, feedback, audit and safety

The discovery profile exposes **25 actions**: the general lab retains its original 19, while
this profile removes implicit sweep/control-generated variants and adds all eight requested
research actions together:

- `hypothesize`: short identified problem, targeted component(s), scientific rationale,
  visible practice ids, metric, nonzero expected effect, confirmation/falsification criteria.
- `inspect_gradients`: opt-in practice-label autograd on one permitted tensor; no optimizer,
  `.grad` accumulation or parameter update. Model/module modes, requires-grad flags and RNG
  are restored. Unsupported backends and allocation/context excesses refuse, not substitute.
- `compare_checkpoints`: original/edited/saved in-memory overlays, not filesystem paths.
- `evaluate_candidate`: paired actual practice accuracy and completion NLL.
- `rollback`: explicit reversal to the previous staged state or original.
- `analyze_failure`, `record_experiment`: link brief effect classifications/revisions to
  immutable measured outcomes; model statements are not causal proof.
- `stop_and_explain`: evidence-limited withdrawal, including after sealing.

Every edit/restore needs a **fresh prospectively committed hypothesis** covering its targets
and current candidate hash. A valid edit automatically receives genuine before/after practice
feedback. A new hypothesis can then revise it; failed or unchanged hypotheses remain in the
journal. Ordered helpful/harmful/noop outcomes, prediction squared errors, repeated numerical
programs and correctly identified harmful trials are reported. Where inspected gradients
match the candidate/cohort, partial first-order alignment is recorded—not semantic relevance.

`edit_journal.json` records **every attempted edit, including rejected commands**: original
checkpoint hash, requested program/numerical values, target shapes/dtypes/hashes, complete
changed flat indices and actual before/after values, rounded changed count and delta norm,
validation/refusal, paired practice accuracy/NLL, reserved outcome if opened, resident
restoration and full-tensor forward/reverse numerical replay checks. Rollbacks are separate
reversal actions, not new independently useful proposals. Fixed follow-up perturbations have
their own complete numerical audit in `retention_measurement.json`.

Completion NLL is teacher-forced reference tokens **plus model EOS**, independently tokenized
after the actual rendered prompt, natural logs, no truncation. Problem-mean and token-mean
loss, token/example/category counts and paired loss differences/SE are distinct from strict
whole-completion NFKC/case/whitespace-normalized accuracy. Cuts/unclosed thinking still fail.
NLL is not a lenient replacement grader or a promotion score.

The study writes its receipt/data/code/config-bound plan **before inference**. A changed
source, model receipt or evaluation input requires a **new** output directory/plan. It
refuses to overwrite or silently resume/retarget a study. The optional in-process measurement
cache binds backend identity, target/candidate state, exact samples, config, grader and seeds;
actual generation/loss counts and cache hits remain separate. Cached repeated answers are
not extra independent evidence. `forward_calls` is the inherited **request** counter;
generated sequences and loss/gradient-evaluated problems are recorded separately.

Each run has 24 turns, four invalid replies, three identical read-only queries before a host
stall, 20 target requests, eight edit actions and a fixed 600-second search budget. Answer
and editor limits are 128 and 512 tokens. Initial diagnosis and sealed/final evaluation are
separate from adaptive request budgets. These are cooperative process limits, not hard
CPU/RAM quotas or an adversarial OS sandbox. A hard kill can prevent integrity proof.

## Selection, generalization and retention

All 44 possible submissions (two checkpoints × two seeds × eleven arms) are included in a
conservative Bonferroni family: **0.05/44 = 0.001136363636… per gate**. Subsetting arms/seeds
does not loosen it. Development and reserved must **both** strictly improve macro accuracy
with zero per-category forgetting and problem-clustered sign-flip p below that threshold.
Equality never passes; loss decreases never override it. This is an engineering preregistration,
not an external timestamp or proof that template-correlated problems satisfy exchangeability.

Submitting seals **before development inference**, including a failed development look;
otherwise development would become more practice. The final selected overlay and last valid
terminal diagnostic overlay are frozen before final inference. Final outcomes never feed
back into this or a later search, rescue rejected candidates or choose a better dose. Cross-arm
comparisons are reported, not used to launch adaptive extra seeds.

The retention split then measures four fixed cells: **base**, **candidate**,
**base+subsequent edit**, **candidate+the same subsequent edit**. Accuracy floors and NLL
retention are reported separately. If no useful edit was accepted, `retention_of_validated_useful_edit`
is **null**, even if a baseline/diagnostic perturbation was tested successfully. This is one
small subsequent edit, not extended continual training or proof of durable meta-learning.

Full JSON is kept in ignored output directories and GitHub artifacts; compact check annotations
include baseline/loss/category/counts, actual edits, outcomes, final/retention and hashes, but
no raw held-out prompts/answers. The [CI workflow](../.github/workflows/ci.yml) runs the longer
360M study on implementation pushes, not duplicate PR or docs-only events. A null scientific
result can pass **integrity CI**; it cannot pass the scientific useful-edit milestone.

## Still missing / not established

- An independently proposed beneficial pretrained edit, unless the actual recorded gates say so.
- A trained editing-policy checkpoint that improves on unseen future editing tasks.
- Broad full-benchmark power, semantic/pretraining decontamination, many checkpoints/seeds,
  stronger clustered/hierarchical treatment of template-related questions.
- Retention after extended further training, and diverse follow-up perturbation families.
- Hard process isolation, arbitrary persistent Python, GPU/vLLM/tensor parallelism and the
  larger paper's full tasks/models. Gradient-assisted controls are **not** no-gradient paper parity.

See [paper parity](PAPER_PARITY.md), [research lab](RESEARCH_LAB.md),
[architecture](SELF_EVOLVING_ARCHITECTURE.md), [MCP](MCP.md) and [testing](TESTING.md).
