# Autonomous Research Foundation

A bounded, single-machine loop that lets one model **propose** training research while an
independent controller, executor and evaluator **measure** it. The proposer cannot score
itself, cannot see hidden data and cannot write the authoritative record.

**Read this first:** this is infrastructure with strict evidence rules, not a demonstrated
self-improving model. The only complete loops executed so far use a **randomly initialised
1M-parameter GPT-2 fixture** on deliberately trivial constant-answer prompts. That exercises
plumbing; it is not a capability, generalization or autonomous-research result. The separate
135M/360M discovery studies remain negative and were **not** restarted or re-purposed.

```bash
ouroboros research tools            # versioned command/authority catalogue
ouroboros research deepseek         # source-linked DeepSeek adopted/deferred matrix
ouroboros research init NEW_DIR --model-path /path/to/local/model
ouroboros research run --config NEW_DIR/foundation.json --check-only
ouroboros research run --config NEW_DIR/foundation.json
ouroboros research verify-run --report NEW_DIR/runs/<run>/research.json
ouroboros research history --database NEW_DIR/experiments.sqlite3
```

`--check-only` validates configuration, data admission and model inventories without loading
a model or creating a database. Real execution needs the `train` extra (Torch/PEFT).

## Roles and authority

| Role | Process | May do | May not do |
| --- | --- | --- | --- |
| Researcher / proposer | fixed `research` worker (or host reference) | emit one versioned JSON proposal of hypotheses + numeric recipes | execute code, read hidden data, set scores, change gates, write the database |
| Controller | trusted parent Python | fix the plan before inference, enforce budgets, decide state, seal selection | invent a recipe when the proposer fails |
| Executor | separate fixed-operation workers | load the target, train, measure, export checkpoints | choose what is trained or when a gate passes |
| Evaluator | **fresh** worker per checkpoint | generate and grade answers on the bound cohort | reuse a warm process, tune on reserved/final data |
| Memory | controller-owned SQLite | append hashed events and host-measured results | accept model narratives as measurements |

The target is an **experimental subject**, not a presumed capable scientist. A stronger
researcher model improving a weaker target is *assisted improvement*; it is not the target
autonomously improving itself, and the report labels which one applies
(`proposal_origin`: `host-authored reference`, `same-target-model`, or
`external-research-model-assisted`).

## The loop

`observe → diagnose → hypothesise → design → train candidates → seal → validation →
reserved → final → robustness → retention → record`.

1. **Observe** (`observe`): baseline practice accuracy/NLL plus bounded introspection
   (architecture fields, parameter metadata, one hidden-state read, one explicitly
   practice-label gradient read).
2. **Diagnose**: observable symptoms only — token-budget exhaustion, empty response,
   incorrect answer, per-domain counts. `causal_claim` is always `false`; answer mismatch
   cannot distinguish data, capacity, format and reasoning failures.
3. **Hypothesise**: 1–4 competing hypotheses, each with confidence, *measured* supporting
   evidence IDs, contradicting evidence, alternatives, missing information, a numeric
   prediction and a falsification condition. Evidence IDs must appear in the measured
   practice context, so a model cannot cite a study it did not observe.
4. **Design**: recipes are type-checked against the immutable budget (steps, families,
   optimizer, schedule, rank/alpha, target modules, domain weights). The design, the ranking
   and the gate set are written **once** (`design.json`) before any candidate is trained.
5. **Train**: each candidate is replicated at every predeclared seed by a fresh worker;
   an exact duplicate of the host control is reused rather than re-measured.
6. **Seal**: one recipe is chosen by **worst-seed practice accuracy** (never a lucky seed),
   and the exact checkpoint list plus manifests are frozen in `sealed_selection.json`
   **before** any validation inference. `future_reselection_allowed: false`.
7. **Gates**: validation → reserved → final → robustness → retention, each requiring a strict
   per-seed accuracy gain, a problem-clustered sign-flip test at `alpha = 0.05 / family_size`
   and zero per-domain forgetting. Robustness is a **non-inferiority** gate.
8. **Retention**: a real fixed follow-up SFT is applied to both the base and each candidate,
   then four cells are compared. This is a regression floor, not an extended-training or
   meta-learning claim.
9. **Record**: terminal state, hashes and the host-measured result go into SQLite; a
   `completed` state requires the full gate chain.

## Data admission

Eight explicit roles are required: `train`, `practice`, `validation`, `reserved`, `final`,
`robustness`, `retention`, `followup`. All must cover the same domains; train/followup labels
must be provenance-verified; duplicates are rejected by normalized prompt identity; every
role file must be a regular non-symlinked file of at most 16 MiB, every example at most
16,000 characters, and screening is bounded to 250,000 words before a model loads.

A **question-overlap screen** (`data_quality.py`) refuses a dataset when a normalized
word n-gram (up to 10 words, minimum 3) from any visible role (`train`, `practice`,
`followup`) occurs in any hidden question. It stops at the first collision and reports the
witness pair. Shared short class labels and answer text are explicitly not screened, and
`semantic_decontamination` is always `false`. This is a narrower, stricter guard than plain
normalized-prompt deduplication — see the
[DeepSeek applicability matrix](../papers/DEEPSEEK_APPLICABILITY.md).

Hidden roles are consumed **before** inference in a cross-run ledger, so a failed or
interrupted look still burns the prompts. Reusing them in a second run fails closed.

## Immutability and provenance

- Model, tokenizer, config and software inventories are captured before loading; the target,
  teacher and researcher model files are all frozen and parameter-capped by preflight.
- `plan.json`, `design.json`, `sealed_selection.json`, `proposal.json`, per-role inputs,
  worker jobs, receipts and per-worker results are **write-once**; a second write raises.
- Each checkpoint carries a manifest (file hashes, run id, recipe, seed, data hash, source
  binding); the controller binds the manifest hash it measured and every evaluator re-checks
  it, so a swapped candidate cannot be scored as the sealed one.
- Completion re-verifies all auxiliary model inventories, the original config/data hashes,
  role snapshots and the software inventory. Any drift fails the run and withdraws promotion.
- `verify-run` re-checks the whole artifact tree, the plan containment and the report's
  binding to the authoritative SQLite result; `--database` supports a relocated ledger.

Integrity is not authentication: hashes detect change, they do not prove authorship, and
they cannot replay inference.

## Execution and resource bounds

Workers are fixed operations (`observe`, `evaluate`, `train`, `research`, `distill`,
`followup`) launched as subprocesses with offline flags, a parent-death `prctl` binding, a
CPU `RLIMIT_CPU`, and parent-side watchdogs for wall time, RSS **and** output bytes. A single
execution lock prevents two controllers owning the same database. Budgets bound per-step
work, total optimizer steps, checkpoint exports, output size and CPU threads; a run that
would exceed the export or step budget is refused *before* launching the worker.

Checkpoints include the optimizer state and CPU RNG as safetensors plus JSON with
`resume_supported: false`. There is **no mid-step resume**: an interrupted run needs a new
plan and fresh independent confirmation.

## Memory and strategy

SQLite holds `baselines`, `experiments`, `events` (hash-chained), `results`, `claims`,
`holdouts`, `prompt_roles` and `candidates` (the counterfactual log of every trained
candidate), with `UPDATE`/`DELETE` triggers that abort. Model claims are
stored with `verified = 0` and never become measurements. `results.useful` requires a
terminal `completed` state **and** `full_gate_passed`.

Strategy selection is transparent, not learned: a Beta–Bernoulli posterior over verified
success per `(context, family)` plus an optimistic success/cost score ranks proposed
recipes. The researcher's history context is restricted to
`context/family/useful/cost` — never prior report bodies, so a model cannot mine earlier
hidden-set outcomes.

An updated posterior is **not** evidence that future experiment selection improved. The
predeclared comparison now exists and is frozen before analysis (`selection.py`, see G6), but
the real log holds far fewer episodes than its predeclared minimum, so it reports
`insufficient_data` rather than a win.

## Status by subsystem

| Subsystem | Status | Note |
| --- | --- | --- |
| Researcher / model intelligence | partial | host reference **and** real local schema-constrained proposer; no demonstrated research capability |
| Architecture/parameter/activation/gradient introspection | implemented (bounded) | practice-only, explicit gradient assistance, no semantic claim |
| Evidence-based diagnosis | implemented | symptoms and uncertainty; `causal_claim: false` |
| Competing hypotheses and ranking | implemented | alternatives, missing information, prediction audit per candidate |
| Immutable controlled designs | implemented | write-once plan/design/sealed selection |
| Multiple intervention types | partial | `lora_sft`, `full_sft`, `data_reweight`, `distill_sft`; architectural edits and audited direct weight edits are outside this loop |
| Owned task data generator | implemented | 5 domains, 8 independent roles, 380 unique prompts, hashed manifest, verification + tamper detection; template fixture, not a benchmark |
| Reproducible training | partial | deterministic flags, seeded order, full traces, optimizer/RNG export; no mid-step resume, no sequence packing |
| Independent evaluation | implemented (this scope) | fresh worker, per-seed, clustered tests, holdout ledger |
| Persistent experiment memory | implemented | SQLite authority + hash-bound JSON/checkpoints |
| Strategy learning | partial | transparent posterior ranking plus a predeclared, hash-frozen selection-policy comparison; real evidence is `insufficient_data` |
| Recovery and safety | partial | state machine, lock, watchdogs, interruption, explicit recovery; **not** an OS sandbox |
| Bounded orchestration | implemented | one loop plus a finite, explicitly capped campaign |

## Evidence status

| Claim | Status |
| --- | --- |
| Infrastructure reproduces a full gated loop | **yes**, on a random tiny GPT-2 fixture with constant answers (32 fresh workers, 208 optimizer steps, 261 verified artifacts, all five gates plus retention, source unchanged) |
| A clean loop runs on a **pretrained** model and owned data | **yes** — 134.5M SmolLM2-135M-Instruct (pinned GGUF → float32) on the owned 380-prompt dataset; 12 fresh workers, 200 optimizer steps, 377 s, peak 1251 MB, source unchanged |
| A host-designed LoRA raises exact-match accuracy on owned data | **measured, not promoted** — practice 0.000 → 0.300 (both seeds), validation 0.000 → 0.267, 16 wrong→right / 0 right→wrong, exact sign-flip p = 0.00195; the predeclared family-corrected alpha 0.05/44 = 0.00114 was *not* met, so the gate rejected |
| The follow-up/retention stages execute on a real model | **yes** (path check, labelled non-evidence): four matched follow-up cells trained and re-scored, retention gate ran to a verdict |
| A host-designed LoRA clears the **validation** gate on a real pretrained model | **yes** — with the power profile: 0.000 → 0.192 pooled, 23 wrong→right, 0 right→wrong, p = 2e-5 |
| A host-designed LoRA clears **every** gate | **no** — that candidate was rejected at the reserved gate: pooled p = 0.00049 but per-seed p = 0.0078 and 0.0020 against α = 0.001136 |
| A model independently proposes a valid, useful experiment | **not established** |
| An autonomous edit survives all gates | **not established** |
| The researcher or selection policy improves with experience | **not established** — no real-data replay yet; the predeclared comparison reports `insufficient_data` on the available log |

The 360M study (22 arms, 64 model replies, 0 numerical edits) and the OOM-interrupted 135M
study are preserved unchanged in [discovery results](DISCOVERY_RESULTS.md); the foundation
does not resume, retarget or reuse their hidden sets.

The observed fixture result is deliberately unimpressive: baseline 0/12 (all twelve answers
truncated), LoRA 0/12 at both seeds, full SFT 12/12 at both seeds, host reference control
0/12. That is memorization of one constant word in one synthetic domain, and the host
hypothesis predicting a +0.01 delta was off by +0.99 — reported, not hidden. See
[testing](TESTING.md#what-the-first-complete-loop-actually-did).

### The owned dataset (G2)

`ouroboros/research/dataset.py` generates the data locally; no third-party question text is
downloaded, and the manifest records the licence (MIT, this repository) so the set is
redistributable.

* 5 exact-answer domains: arithmetic, comparison, unit conversion, digit sequences and
  divisibility (yes/no, so answers are never always numeric).
* 8 roles with their own salted ordinal blocks:
  train 150, practice 30, validation 30, reserved 40, final 40, robustness 30, retention 30,
  followup 30 — **380 prompts, globally unique**, so a training item can never double as
  hidden evidence.
* Every sample passes the same `ResearchData` admission screen the controller applies, and the
  question-overlap screen ran 1991 normalized 10-word checks with **0 collisions** between the
  visible roles (train/practice/followup) and the hidden ones.
* `manifest.json` carries per-file hashes, the dataset fingerprint
  (`6351b718…c32bbb` for seed 17), the overlap report and the non-claims;
  `verify_dataset()` re-hashes everything and refuses a changed file.

Non-claims, repeated because they matter: this is a **template-generated diagnostic fixture**,
not a benchmark, not comparable to GSM8K/MATH/IFEval, not decontaminated against any model's
pretraining corpus, and a gain on it is not a reasoning or generalization result. The `Item N.`
ordinal prefix is fixture scaffolding that keeps prompts disjoint — it also means a model could
in principle key on it, which is one more reason the numbers above are diagnostics only.

### The clean pretrained run (G1)

The run below replaces the earlier attempt that died on a worker contract bug. It used the
pinned pretrained source in `ouroboros/model_files.py` (`simonw/llm-smollm2`, revision
`afed0963…`, GGUF sha256 `b179c952…10db53`, Apache-2.0) dequantised to float32 safetensors by
`fetch_smoke_model`, and the owned dataset described under G2. Numbers come from
`outputs/g1-owned-01/runs/research-3c10902ee2654d289b79c51adbee5dc1/research.json`; that tree
is git-ignored, so the record lives here.

| Stage | Baseline | LoRA SFT (r=4, α=8, lr 2e-4, 50 steps) |
| --- | --- | --- |
| Practice macro-accuracy (30 problems) | 0.000 | 0.300 (seed 17) / 0.300 (seed 18) |
| Validation (30 problems, 2 answers each) | 0.000 | 0.233 (p 0.0078) / 0.300 (p 0.00195) |
| Pooled validation (both seeds) | 0.000 | 0.267, 16 wrong→right, 0 right→wrong, p = 0.001953 |
| Predeclared gate | — | **rejected**: 0.001953 > α = 0.05/44 = 0.001136 |

The second candidate family (`data_reweight`) scored 0.267/0.300 on practice; the sealed
selection rule (worst-seed practice accuracy) chose `lora_sft`. Nothing was promoted, no
checkpoint was applied to the baseline, and the rejection stands exactly as recorded.

Read this honestly. The baseline is at the floor: it answers 0.000 because a 135M instruct
model rarely emits the exact short string in 12 new tokens, so the measured gain is mostly
*task and answer-format acquisition*, not reasoning. The effect is real (exact one-sided
sign-flip test over problems, zero right→wrong) but it does not clear the family-corrected
threshold fixed before the run, which is what the gate is for. With 30 validation problems
only 9 differ, and the exact test bottoms out at 2⁻⁹ = 0.00195 — so at this dataset size a
candidate has to win on more problems, not just win bigger, to pass α ≈ 0.00114.

### The follow-up/retention path (G1 continuation)

Because the candidate was rejected at validation, the controller never entered the follow-up
stage. `scripts/run_followup_path_check.py` therefore replays the four-cell retention design
on the recorded run so the downstream code path is exercised with real training:
before base 0.000, before candidates 0.067/0.133; after the fixed follow-up SFT 0.000/0.000
base and 0.067/0.133 candidate; `retention_gate` returned `passed: true` on the
no-regression floors in 262 s. The report is written with
`scientific_evidence: false` and explicit non-claims (it replays a rejected candidate, adds no
prompts and cannot promote anything). Writing it caught a real requirement: every retention
measurement must use one fixed evaluation seed, otherwise the paired guard refuses the
comparison — that guard fired and was respected, not bypassed.

### Test power, and the first candidate to reach the reserved gate (G3, G7)

The G1 run could not pass validation for a structural reason: with 30 validation problems only
9 separated the arms, and an exact one-sided sign-flip test over 9 problems bottoms out at
2⁻⁹ = 0.00195 — above the predeclared α = 0.05/44 = 0.00114. The `power` dataset profile
therefore enlarges the *hidden* roles (validation/reserved/final 60, robustness/retention 40,
practice 40) and leaves the training role at the size the recipe was tuned for. Two power runs
were executed end to end:

| Run | Validation (60 problems, 2 answers each) | Reserved (60 problems) | Outcome |
| --- | --- | --- | --- |
| power-02, seed 18, training role 300 | 0.000 → 0.100 / 0.067; 6 and 4 separating problems; p 0.0156 / 0.0625 | not reached | rejected at validation |
| power-03, seed 19, training role 150 | **passed**: 0.000 → 0.192 pooled; 23 wrong→right, 0 right→wrong; p = 2e-5 (per seed 0.00024 and 0.00049) | **failed**: 0.000 → 0.133 pooled; 16 wrong→right, 0 right→wrong; pooled p = 0.00049 but per-seed p = 0.0078 / 0.0020 | **rejected at reserved** |

Three things are actually established by that table, and one is not:

1. **Power was the blocker, not the recipe.** Same recipe family, same model: 23 separating
   problems at 60 validation problems give p = 2e-5, while 9 at 30 problems give 0.00195.
2. **More training data at a fixed step count is not more learning.** Doubling the training
   role to 300 while keeping 50 optimizer steps under-fit the model (practice 0.225/0.200
   against 0.300 at 150 examples), so the run failed at validation. The profile was corrected
   back to 150 rather than quietly raising the step budget after seeing the result.
3. **The reserved gate is stricter in practice than validation.** The pooled reserved effect
   stayed significant (p = 0.00049) but the per-seed tests did not (0.0078 and 0.0020 versus
   α = 0.001136). Because the gate demands reproducibility at every predeclared seed, the
   candidate was rejected. This is the first real pretrained candidate to clear validation and
   the gate still refused to promote it.
4. **What is *not* established:** that the model learned the tasks. Its gain is domain-narrow —
   divisibility 0.667 and comparison 0.458, with arithmetic, conversion and sequence at 0.000 —
   i.e. the two short-answer formats, not reasoning. A candidate that passes every gate on this
   fixture would still be a diagnostic, not a capability.

For G7 that is a characterized power statement rather than a solved one: to clear
α = 0.001136 an exact per-seed sign-flip test needs **at least 10 separating problems per
seed**, and the observed counts (7 vs 9, 11 vs 12) show the variance is wide enough that a
60-problem hidden role is the minimum, not a comfortable margin. More seeds and more hidden
problems remain the honest next step.

### Selection-policy comparison (G6)

`ouroboros/research/selection.py` freezes the design *before* it looks at any outcome:
policies `proposal_order` / `posterior_optimistic` (primary) / `random_fixed_seed`, one-sided
paired sign-flip test over episodes, α = 0.05, minimum 20 episodes, chronological episode
order, information set restricted to strictly earlier episodes, no online learning, no trained
researcher, no hidden data. The design is written once to
`outputs/g1-owned-01/selection/selection_design.json` (sha256 `23505c2a…`) and cannot be
rewritten.

To make the replay possible at all, the controller now logs **every** trained candidate, not
only the winner, into a `candidates` table (ordinal, family, recipe hash, practice accuracy,
baseline accuracy, `gate_passed`, `selected`). The G1 run logged its two candidates. That is
**one** episode against a predeclared minimum of 20, so the honest output is:

```text
status: insufficient_data   episodes: 1   required: 20
improved_future_selection_demonstrated: false
```

No policy claim is made on real data. The machinery itself is checked two ways, both labelled
non-evidence: a synthetic 48–60 episode fixture with a known family advantage yields
`policy_better_on_replay` at p ≈ 1.2e-4 with the null policy never winning, an ambiguous
fixture stays `not_established`, and a unit test replays 24 logged episodes through the real
SQLite log. Reaching the predeclared threshold needs ≥ 20 real episodes, i.e. ≥ 20 foundation
runs on fresh datasets (roughly two hours of CPU at the settings above); that is a data
collection cost, not a code change, and it is listed as the remaining G6 work.

## Reproduction

```bash
# Protocol-only (no Torch): contracts, admission, gates, memory, dataset, selection CLI
python -m pytest -q tests/test_research_foundation.py tests/test_research_data_quality.py \
  tests/test_research_dataset_selection.py

# Real offline CPU loop on a random fixture; never a pretrained claim
python scripts/run_foundation_fixture.py --workdir outputs/foundation-fixture-01

# Pinned pretrained source (GitHub only, hash-verified): 98 MB GGUF -> float32 safetensors
python -c "from ouroboros.model_files import fetch_smoke_model; \
print(fetch_smoke_model('models/smollm2-135m-instruct-gguf')['verified'])"

# Owned dataset + a clean pretrained run on a brand-new directory
python -m ouroboros.cli research data --directory outputs/g1-owned-01 \
  --model-path models/smollm2-135m-instruct-gguf
python -m ouroboros.cli research data --directory outputs/g1-owned-01 --verify
python -m ouroboros.cli research run --config outputs/g1-owned-01/foundation.json
python -m ouroboros.cli research verify-run --report outputs/g1-owned-01/runs/*/research.json

# Predeclared selection comparison: real log first, synthetic machinery check second
python -m ouroboros.cli research compare-selection \
  --database outputs/g1-owned-01/experiments.sqlite3 --output outputs/g1-owned-01/selection
python -m ouroboros.cli research compare-selection --fixture   # labelled non-evidence

# Follow-up/retention path check on the recorded run (real training, non-evidence)
python scripts/run_followup_path_check.py
```

`scripts/run_foundation_fixture.py` creates a new directory, writes a 1M random model plus a
constant-answer scaffold, runs the loop, verifies the artifact tree and writes
`fixture_summary.json` with explicit non-claims.

Environment note: in a 4 GB cgroup `pip install '.[train,llm,smoke]'` is killed (exit 137)
while resolving the 906 MB Torch/CUDA bundle. The working recipe used here is
`pip download --no-deps torch==2.5.1`, `pip install --no-deps --no-cache-dir` of that wheel and
of each `nvidia-*-cu12` runtime library in its own invocation, then the pinned
`transformers`/`peft`/`accelerate`/`tokenizers`/`huggingface-hub`/`gguf` wheels with
`--no-deps` plus their small pure-Python dependencies. Version-skew warnings between the
Torch-pinned CUDA 12.4 libraries and newer 12.9 packages are benign; missing `.so` files are
not.

## Open gaps

Every item below is missing *now*. IDs are stable so the gap register can be cited from
issues or a changelog entry.

### Scientific (blocking any capability claim)

| ID | Gap |
| --- | --- |
| G1 | **Closed for the plumbing, negative for promotion.** A clean pretrained run now completes on owned data (numbers above) and the follow-up/retention path has executed. What remains is *scientific*: with 30 validation problems the exact test cannot go below 2⁻⁹ unless more problems separate the arms, so the α = 0.05/44 gate needs a bigger validation role (G7) before a candidate can pass. |
| G2 | **Closed, with limits stated.** The owned dataset exists (380 unique prompts, 5 domains, 8 independent roles, hashed manifest, verification and tamper detection) and is used by the run above. It is template-generated and not a benchmark; a semantically harder task family is still missing (G3). |
| G3 | **Open, but past the first gate.** A host-designed LoRA now clears the validation gate on a real pretrained model (0.000 → 0.192, p = 2e-5) and was then rejected at the reserved gate on per-seed reproducibility (0.0078 / 0.0020 vs α = 0.001136) despite a significant pooled effect. Its gain is domain-narrow (two short-answer domains), so no capability claim follows even if a later candidate passes. |
| G4 | **No independent model proposal of demonstrated value.** The local proposer path exists but has never produced a proposal that survived training and gating on a real task. |
| G5 | **No useful autonomous edit** under the full gate chain (validation → reserved → final → robustness → retention). |
| G6 | **Machinery closed, evidence open.** The comparison is predeclared, frozen and hash-bound, and the counterfactual candidate log now exists. The real log holds 1 episode against a predeclared minimum of 20, so the reported status is `insufficient_data`; no policy claim is made. Collecting ≥ 20 real episodes is the remaining work. |
| G7 | **Power is characterized, replication is still thin.** With the power profile the exact per-seed test needs ≥ 10 separating problems to beat α = 0.001136; observed counts were 7 vs 9 and 11 vs 12, and doubling the training role at a fixed step budget hurt. Two seeds and one machine remain: more seeds and larger hidden roles are the next step, and no variance estimate over seeds should be quoted from two of them. |

### Engineering (implemented but bounded)

| ID | Gap |
| --- | --- |
| G8 | **No OS sandbox.** Workers are subprocesses with offline flags and watchdogs; hidden files, the ledger and the output tree stay readable by any local process. |
| G9 | **Cooperative budgets only.** Wall/RSS/output-byte/step/export limits are checked by the controller, not enforced by cgroups or quotas. |
| G10 | **No mid-step resume.** Optimizer/RNG state is exported with `resume_supported: false`; an interrupted run needs a new plan and fresh confirmation. |
| G11 | **No intermediate checkpoint-selection protocol.** Only the final checkpoint of each run is evaluated; epoch/step selection with its own holdout is unimplemented. |
| G12 | **No sequence packing**, no gradient accumulation tuning, no mixed precision, no distributed/multi-GPU execution. |
| G13 | **No architecture or parameter-count intervention** and **no audited direct weight edit** inside this loop (the lab/discovery paths handle weight edits separately, and are not wired to these gates). |
| G14 | **No learned researcher or bandit policy.** Ranking is still a transparent posterior/cost heuristic. The *counterfactual logging* half of this gap is now done: every trained candidate is recorded (`candidates`), so proposals can be replayed offline (G6). What remains is actual policy training — that is deliberate, since a trained policy needs many more episodes than exist (G6) and would otherwise be fitted to noise. |
| G15 | **No research MCP tools.** The foundation is CLI/API only; MCP exposure would need new approval bindings. |
| G16 | **Integrity is not authentication.** Hashes detect edits; there is no signing, no external timestamping, no inference replay. |
| G17 | **Data admission is not decontamination.** The screen covers normalized question n-grams only; answers, semantics and pretraining contamination are explicitly unknown. |
| G18 | **No retention under extended training.** Retention uses a fixed tiny follow-up SFT, not continued training or meta-learning. Closing it needs a declared, budgeted continued-training schedule and a holdout that survives several such schedules — none of the current roles are sized for that. |

Why the remaining engineering gaps are not simply "more work on this machine": G8 and G9 need
OS-level isolation (namespaces/cgroups) that a subprocess budget check cannot provide; G10 is a
deliberate refusal, not a backlog item — an interrupted run must start a new plan rather than
resume mid-step, because a resumable step is a hidden-truncation risk; G11 needs a
checkpoint-selection protocol with its own untouched holdout, which the current roles are not
sized for; G12 needs GPU hardware (mixed precision and sequence packing are cheap to add, but
they buy nothing on a two-thread CPU run); G13 needs the lab/discovery weight-edit path wired
into *these* gates, which is a separate integration with its own evidence; G15 needs
approval-bound MCP tools rather than read-only mirrors of the CLI; G16 needs a signing key and
an external timestamp, neither of which exists here; G17 needs a real contamination corpus to
screen against (the current screen is bounded question n-grams, and G2's data is generated, so
there is nothing to decontaminate it from); G18 needs a longer training schedule and holds big
enough to survive several of them.

### Scope (deliberately out)

| ID | Gap |
| --- | --- |
| G19 | **RL/GRPO.** No policy-gradient objective, reward model or RL infrastructure; group standardisation is a data-filtering diagnostic. |
| G20 | **Formal verification.** No Lean/proof-assistant integration, so Prover-style verification is limited to exact-answer matching. |
| G21 | **Learned verifiers / self-scoring.** Deliberately refused; grading stays exact-answer and host-owned. |
| G22 | **MoE, MLA, sparse attention, Engram-style model memory, multi-token prediction.** Architecture work, not post-hoc interventions. |
| G23 | **Multimodal** (VL, VL2, Janus, OCR, OCR-2). No encoders, image data or visual evaluators. |
| G24 | **Fill-in-the-middle and repository-level code mining.** Completion-style SFT only. |
| G25 | **Paper-scale reproduction.** No DeepSeek weights, corpora, token counts or reported benchmark numbers are reproduced anywhere in this repository. |

### Evidence hygiene

| ID | Gap |
| --- | --- |
| G26 | **Frozen 135M arm remains interrupted** with unverified resident restoration; it was never resumed, and its eighth arm stays incomplete. |
| G27 | **The CI 360M reruns** execute the protocol on fresh runners; they are not independent replications of the frozen local studies and must not be reported as such. |
| G28 | **PDF review was partial.** Equations/figures were lost in text extraction and some batched output was truncated, so mechanism descriptions may be coarser than the papers. |

## Working direction

Long-term direction **D — continuous bounded development** — with measurable milestones, each
gated on its own evidence:

- **A** reproducible single loop (achieved as a fixture; real-data rerun pending)
- **B** a host-designed intervention that benefits a real pretrained model
- **C** independently generated proposals with strictly valid features (not "plausible text")
- **D** a useful autonomous edit surviving every independent gate
- **E** a predeclared demonstration that experience improves future experiment selection

Nothing above B is claimed.
