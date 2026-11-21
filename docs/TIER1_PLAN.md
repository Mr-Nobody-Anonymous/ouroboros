# Tier-1 plan: how each open scientific gap is designed, measured and closed

The register in [research foundation](RESEARCH_FOUNDATION.md#open-gaps) names four gaps that block
any capability claim: **G3** (a task family where the gain is not saturation), **G4** (a proposer
that emits a valid, useful recipe), **G5** (the lab/discovery edit path wired into these gates)
and **G6** (a discriminating selection design). This document is the design record for all four:
what was built, what was predeclared, what was measured, and what would falsify each claim.

Every item obeys the same four rules, because they are what makes the results checkable:

1. **Predeclare, then run.** A design is frozen (hash recorded) before data is collected. A design
   changed after seeing outcomes is a new design with a new hash — never a re-tune of the old one.
2. **Fresh data only.** Consumed holdouts are never replayed, relabelled or resumed. A failed run
   is diagnosed and re-run on a new seed.
3. **The null is a result.** A measurement that does not separate arms is reported as
   `not_established` with its mechanism, not softened into a trend.
4. **The repository is the source of truth.** External tools (Linear, Sheets, Notion) mirror the
   register; a status change alone never closes a gap.

---

## G3 — a task family where the gain is not saturation

**Problem.** Every candidate that passed the five gates did so on template domains that a 50-step
LoRA saturates (practice 1.000). "Passed all gates" therefore demonstrated plumbing, not learning.
What is missing is a family where the model learns *part* of the task and the split decides whether
it transfers.

**Research input.** *Teaching Arithmetic to Small Transformers* (Lee et al., 2023) measures a sharp
phase transition for two-digit addition near ~1,500 balanced training samples, and shows that
random sampling fails where balanced sampling succeeds; length generalisation does not appear from
same-length training. At this budget (≤ 480 training examples, 50–150 optimizer steps) a two-digit
*addition* task is therefore expected to sit well below ceiling — which is exactly what makes a
carry split informative rather than decorative.

**Design (`staged` profile, 560 prompts).** Two-digit plus one-digit addition, split by the event
that makes the task compositional — whether the units column carries:

| role | domains | size |
| --- | --- | --- |
| train / followup | `carry_add` only (units sum ≤ 9, **never** carries) | 160 / 40 |
| practice | both halves | 40 |
| validation, reserved, final | both halves | 80 each |
| robustness, retention | both halves | 40 each |

* The operand space is large (`tens 1–9 × units 0–9 × one-digit addend`), so visible and hidden
  prompts share no wording: the admission screen's 10-word overlap check returns zero collisions,
  and each role draws its pairs from a hash scatter rather than a cycling index.
* The *answer function* is identical across halves; only the carry event differs. A candidate that
  memorises "units never carry" scores on the visible half and near chance on the held-out half.
* Admission now permits an evaluation role to be **wider** than training (a declared holdout) and
  still refuses any role that is **narrower** — a hidden role missing a domain could hide a
  failure. The split is written into `manifest.json` and into the run plan (`domain_split`).

**Measurement.** Per-half accuracy is reported for the baseline and the candidate on every gate,
so the result is a 2×2 (in-distribution vs held-out) × (before vs after) table, not one number.
The gates themselves are unchanged (α = 0.05/44, per-seed reproducibility, zero-domain forgetting).

**Falsification / acceptance.**
* *Accepted* only if the candidate improves the **held-out** half at every seed, below saturation.
* A gain confined to the in-distribution half is a **negative** result about transfer and is
  reported as such.
* A flat zero on both halves means the family is beyond the budget at this model size — also a
  result, and it moves the next step to more steps or a smaller task, not to a wider claim.

**First execution — done (`outputs/g3-staged-01`, seed 501).**

| quantity | baseline | candidate |
| --- | --- | --- |
| practice, in-distribution | 0.000 | 0.250 / 0.300 (two seeds) |
| practice, **held-out carry** | 0.000 | **0.000 / 0.000** |
| validation / reserved / final (pooled) | 0.000 | 0.219 / 0.175 / 0.163 |
| held-out half at every gate | 0.000 | **0.000** |
| truncation rate, validation | 63.75 % | 0.00 % |

`state: completed`, `full_gate_passed: true`, 1201 s. The candidate learned the answer format and
partial in-distribution arithmetic, and transferred **nothing** to the composition it never saw.
Verdict: **negative for transfer, positive for measurement** — the family is genuinely
non-saturated (0.33–0.53, never 1.000) and the split discriminates.

**Gate consequence (declared, not retro-applied).** The pooled gates promoted a candidate with a
0.000 held-out half, because "no domain worse than baseline" is satisfied by 0.000. The next
predeclared G3 design therefore adds one criterion *before* running: a candidate claiming
compositional transfer must show a positive held-out delta at every seed. It is not applied to
`g3-staged-01`, whose record stays as measured.

---

## G4 — a proposer that emits a valid, useful recipe

**Problem.** The 135M proposer exhausted a 256-token budget inside one schema slot and degenerated
into a digit run; the controller correctly abstained. The failure is in the *interface*, not the
controller: the full proposal schema (recipe + hypotheses + alternatives + falsification +
predictions) is larger than a 135M model can fill in one constrained pass.

**Research input.** Constrained decoding guarantees *shape*, never truncation-safety or
correctness, and it is known to trap small models in structural repetition (the "grammar-constrained
repetition trap"); recommended practice is deterministic repair for format failures plus a short,
draft-conditioned re-shape pass, with escalation only for *content* failures
([structured-output practice](https://dev.to/libme/llm-returns-invalid-json-once-in-a-while-structured-outputs-vs-validate-and-retry-5faj),
[format-failure ladder](https://getnadir.com/blog/format-failures-small-model-structured-output-escalation-cost/)).

**Design (three layers, each independently auditable).**

1. **Short schema (`proposal_v2`).** One recipe, one hypothesis, no nested alternatives:
   `{choice, family, steps, learning_rate, rank, target_modules, hypothesis:{claim, confidence},
   evidence_ids, rationale}` — roughly a third of the current token count.
2. **Draft-then-transcribe.** The model first writes two free-form lines ("what I would try, why"),
   then a *short* constrained pass transcribes that draft into `proposal_v2`. Transcription is the
   easy task for a small model; the reasoning is not squeezed through the grammar.
3. **Host-owned deterministic repair.** Only *syntax* is repaired (fences, trailing commas, an
   unterminated object closed at the last balanced point). Every repair is logged with before/after
   text and a `repaired: true` flag; **no value is invented, no field is filled in**, and a schema
   violation stays a violation. The controller still abstains on anything invalid, and still never
   substitutes a host recipe for model output.

**Built (all three layers, in-tree).**

* `proposal_schema_v2()` — 788 characters against the v1 schema's several thousand; eight required
  fields, two enum-constrained module names.
* `local_proposal_v2()` — optional draft stage, then the constrained short transcription; the draft
  can be switched off (`researcher.draft: false`) as a declared A/B.
* `repair_json()` — deterministic and syntax-only: fences/prose dropped, trailing commas removed, an
  unterminated tail cut at the last balanced point and reported `truncated_unbalanced` (never
  silently "fixed" by inventing structure).
* `host_filled_fields` — the short schema omits `alpha`, `optimizer`, `schedule`, `weight_decay`,
  `hypothesis_index` and `domain_weights`; the host fills them at declared defaults and every
  filled field is logged, so the model's choice and the host's defaults never blur together.

**First executions.**

| run | seeds | interface | outcome |
| --- | --- | --- | --- |
| `g4-local-02` (v1) | 302 | full schema, 256 tokens | truncated inside a slot, degenerated into a digit run, abstained |
| `g4-local-03` (v2, draft) | 303 | short schema, draft → transcribe | **valid JSON in 111 tokens, `finish_reason: eos`, zero repairs** — but the free-form draft looped ("The training data contains…" ×96 tokens) and the transcription faithfully copied `abstain` from it |
| `g4-local-04` (v2, no draft) | 304 | short schema, direct | digits ran to exhaustion inside `confidence`; budget exhausted |
| `g4-local-05` (v3) | 305 | every number an enum | valid, `eos`, `abstain` at `confidence_percent: 0` |
| `g4-local-06` (v3, minimal context) | 306 | enum numerics, small context | chose **`experiment`** (`distill_sft`, 100 steps, `lr_1e4`, rank 2) then degenerated inside the free-text `claim` |
| `g4-local-07` (v4) | 307 | claim is a declared choice | **complete valid proposal, 86 tokens, 0 repairs** — rejected by the host for exceeding the 60-step budget |
| `g4-local-08` (v4, derived menu) | 308 | step menu from the budget | valid, `eos`, `abstain` at `confidence_percent: 0` |

The progression is the finding: each iteration was forced by a *measured* failure, and the
remaining failure is no longer serialisation. The frozen handoff interface for a stronger
researcher is `schema: v4`, `context: minimal`, `draft: false`.

So the *interface* half of G4 is closed — `g4-local-07` is the proof — and the *capability* half is
not: the same interface flips between `experiment` and `abstain` with the context and always
reports confidence 0. The three rules that closed the interface half are now design requirements,
not options: numeric fields are enums, free text is a choice among declared claims, and menus are
derived from the budget.

**Measurement.** A proposal is *valid* if it parses, validates against the budget and the evidence
IDs, and names a supported target module. It is *useful* only if the controller subsequently trains
it and the gates decide it — usefulness is never asserted by the proposer.

**Falsification / acceptance.** Success = a real run where `proposal_origin: model`, `valid: true`,
`fallback_used: false`, and the candidate is actually trained. A valid-but-useless proposal is a
partial result and is reported as such; an invalid proposal with a repair log is a *better*
negative result than a silent abstention, because it shows where the model failed.

---

## G5 — the lab/discovery edit path wired into these gates

**Problem.** The gates, seal, ledger, rollback and promotion machinery all exist and have run. The
edit path (weight-space / architectural edits, its own evaluator and data) does not submit
candidates to them. Connecting the two is milestone D.

**Design (authority separation first, plumbing second).**

| authority | may do | may not |
| --- | --- | --- |
| researcher | propose a recipe or an edit *description* | apply an edit, score it, choose data |
| editor (lab path) | apply a described edit to a checkpoint, produce a new artifact + manifest | see hidden roles, set scores, write the ledger |
| host controller | train, evaluate, seal, gate, promote, write the ledger | invent a proposal, relax a gate |
| gate | decide from hidden-role measurements | be re-run after seeing the outcome |

An edit becomes a **candidate** by producing the same object a trained recipe produces: a
hash-bound checkpoint manifest plus a declared provenance record (`edit_kind`, `applied_to`,
`base_sha256`, `described_by`). The gate path then treats it exactly like a trained candidate —
same practice-only seal, same five gates, same promotion and rollback. Two properties must hold and
are testable without a real edit: (a) an edit candidate is rejected if its manifest does not match
its bytes; (b) an edit that was never described by the researcher cannot enter the chain
(`described_by` binding).

**Why it is last.** It needs a fresh data family (the consumed 360M/135M prompts and the `g3-*`
hidden roles are spent), and it is the only gap where a *wrong* wiring could silently let an
unsealed artifact be scored on hidden data. Both risks are protocol risks, not compute risks, so
this one is designed and fixture-tested before any real run.

---

## G6 — a discriminating selection design

**Problem.** The frozen design scores a proposal by the binary event "did the candidate beat the
baseline at this seed?". Both families beat it in 20/20 episodes, so every comparison tied and the
exact sign-flip test returned p = 1.0. The null is **structural**: the statistic has no resolution
left on this log.

**Design (`magnitude_v2`, new hash).** Same policy arms, same episode ledger, different statistic:

* an episode's score is the **practice accuracy gain** of the chosen candidate over the baseline
  (a magnitude in [0, 1], already logged for every candidate), not a win/loss bit;
* the paired comparison remains exact and distribution-free: the sign-flip test is applied to
  *signed magnitudes* (primary − null, per episode), and the decision rule requires both a positive
  median and a p below the family-corrected α;
* ties are broken by the declared magnitude noise floor ε (an episode whose |Δ| < ε contributes 0),
  which is fixed **before** the replay and reported with the result.

Because this changes the statistic, it is a new design: new module constant, new design hash, and a
new replay directory. The old design stays frozen and its result stays `not_established` — the
magnitude design does not retro-fit a win into the old log, and the first replay on the existing 20
episodes is labelled **exploratory re-analysis** (the log has been seen). Confirmation requires new
episodes collected after the design is frozen.

**Built so far.** `design_magnitude()` (hash `f166076cfc02432e…`, distinct from v1's
`23505c2aefd0404d…`), `magnitude_ucb` arm policy with forced exploration, `compare_magnitude()`
and `analyse_magnitude()`, reachable through `research compare-selection --design magnitude_v2`.
Unit-tested on a synthetic log where both arms always improve (the exact shape that made v1 tie):
v2 reports 0 episodes worse, ≥ half better, positive median, `policy_better_on_replay`.

**Not yet run on real episodes.** The v1 campaign ledger was consumed by a sandbox reset, so the
first real replay needs ~20 fresh `compact` episodes. Until that replay exists the design is
*frozen and validated, not evidenced* — and the writeup says so.

**Falsification / acceptance.** `policy_better_on_replay` requires the magnitude comparison to
separate the arms at α with a positive median; anything else is reported as `not_established` with
the observed median, the sign counts and the noise floor.

---

## Sequencing and cost

| order | gap | why this order | wall time |
| --- | --- | --- | --- |
| 1 | G6 | cheapest; turns a dead null into a test with resolution | ~25 min (replay) + episodes |
| 2 | G3 | the only one that can turn plumbing into evidence | ~25 min per run |
| 3 | G4 | unblocks genuine autonomy; interface work, measurable in one run | ~15 min per run |
| 4 | G5 | highest protocol risk; design + fixture proof before any real edit | days, not minutes |

Items G7–G18 remain blocked on capabilities this machine does not have (OS isolation, signing,
GPU, a contamination corpus, longer schedules); their per-item blockers are unchanged in the
register and are not re-litigated here.

## Sources consulted for the designs

* Lee et al., *Teaching Arithmetic to Small Transformers* (arXiv:2307.03381) — phase transition
  near 1.5k balanced samples; random vs balanced sampling; no length generalisation.
* Patiorta, *Arbitrary-Length Generalization for Addition in a Tiny Transformer* (arXiv:2406.00075)
  — carry-aware instance design.
* Structured-output practice: constrained decoding guarantees shape only; deterministic repair for
  format, short draft-conditioned re-shape for small models, escalation only for content failures.
