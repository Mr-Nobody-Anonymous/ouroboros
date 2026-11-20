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
`holdouts` and `prompt_roles`, with `UPDATE`/`DELETE` triggers that abort. Model claims are
stored with `verified = 0` and never become measurements. `results.useful` requires a
terminal `completed` state **and** `full_gate_passed`.

Strategy selection is transparent, not learned: a Beta–Bernoulli posterior over verified
success per `(context, family)` plus an optimistic success/cost score ranks proposed
recipes. The researcher's history context is restricted to
`context/family/useful/cost` — never prior report bodies, so a model cannot mine earlier
hidden-set outcomes.

An updated posterior is **not** evidence that future experiment selection improved. That
milestone needs a predeclared comparison of selection policies on fresh tasks and has not
been run.

## Status by subsystem

| Subsystem | Status | Note |
| --- | --- | --- |
| Researcher / model intelligence | partial | host reference **and** real local schema-constrained proposer; no demonstrated research capability |
| Architecture/parameter/activation/gradient introspection | implemented (bounded) | practice-only, explicit gradient assistance, no semantic claim |
| Evidence-based diagnosis | implemented | symptoms and uncertainty; `causal_claim: false` |
| Competing hypotheses and ranking | implemented | alternatives, missing information, prediction audit per candidate |
| Immutable controlled designs | implemented | write-once plan/design/sealed selection |
| Multiple intervention types | partial | `lora_sft`, `full_sft`, `data_reweight`, `distill_sft`; architectural edits and audited direct weight edits are outside this loop |
| Reproducible training | partial | deterministic flags, seeded order, full traces, optimizer/RNG export; no mid-step resume, no sequence packing |
| Independent evaluation | implemented (this scope) | fresh worker, per-seed, clustered tests, holdout ledger |
| Persistent experiment memory | implemented | SQLite authority + hash-bound JSON/checkpoints |
| Strategy learning | partial | transparent posterior/bandit ranking only; improvement not demonstrated |
| Recovery and safety | partial | state machine, lock, watchdogs, interruption, explicit recovery; **not** an OS sandbox |
| Bounded orchestration | implemented | one loop plus a finite, explicitly capped campaign |

## Evidence status

| Claim | Status |
| --- | --- |
| Infrastructure reproduces a full gated loop | **yes**, on a random tiny GPT-2 fixture with constant answers (32 fresh workers, 208 optimizer steps, 261 verified artifacts, all five gates plus retention, source unchanged) |
| A host-designed control benefits a real pretrained model | **not established** |
| A model independently proposes a valid, useful experiment | **not established** |
| An autonomous edit survives all gates | **not established** |
| The researcher or selection policy improves with experience | **not established** |

The 360M study (22 arms, 64 model replies, 0 numerical edits) and the OOM-interrupted 135M
study are preserved unchanged in [discovery results](DISCOVERY_RESULTS.md); the foundation
does not resume, retarget or reuse their hidden sets.

The observed fixture result is deliberately unimpressive: baseline 0/12 (all twelve answers
truncated), LoRA 0/12 at both seeds, full SFT 12/12 at both seeds, host reference control
0/12. That is memorization of one constant word in one synthetic domain, and the host
hypothesis predicting a +0.01 delta was off by +0.99 — reported, not hidden. See
[testing](TESTING.md#what-the-first-complete-loop-actually-did).

A first real-model loop attempt reached the follow-up stage and failed on a worker contract
bug (the training worker read a retention measurement it must not see). That is fixed: the
follow-up worker now returns only its training record plus
`evaluation_requires_fresh_worker: true`, and post-follow-up scoring happens in a separate
evaluator process. A clean **pretrained** rerun is still required before any result on a real
model is quoted.

## Reproduction

```bash
# Protocol-only (no Torch): contracts, admission, gates, memory, CLI
python -m pytest -q tests/test_research_foundation.py tests/test_research_data_quality.py

# Real offline CPU loop on a random fixture; never a pretrained claim
python scripts/run_foundation_fixture.py --workdir outputs/foundation-fixture-01
```

`scripts/run_foundation_fixture.py` creates a new directory, writes a 1M random model plus a
constant-answer scaffold, runs the loop, verifies the artifact tree and writes
`fixture_summary.json` with explicit non-claims.

## Limits

- Sequential CPU workers; no distributed training, multi-GPU or multi-agent concurrency.
- Process separation is not filesystem secrecy: the controller is trusted Python, and a
  hostile local reader could still see hidden data files.
- Cooperative budgets are not hard quotas; there is no adversarial model-code sandbox.
- No architectural/parameter-count intervention, no audited direct weight edit inside this
  loop, no intermediate checkpoint-selection protocol, and no learned researcher policy.
- Multimodal, MoE/MLA, sparse-attention and formal-proof features are out of scope and are
  individually listed, with reasons, in [the DeepSeek matrix](../papers/DEEPSEEK_APPLICABILITY.md).

## Working direction

Long-term direction **D — continuous bounded development** — with measurable milestones, each
gated on its own evidence:

- **A** reproducible single loop (achieved as a fixture; real-data rerun pending)
- **B** a host-designed intervention that benefits a real pretrained model
- **C** independently generated proposals with strictly valid features (not "plausible text")
- **D** a useful autonomous edit surviving every independent gate
- **E** a predeclared demonstration that experience improves future experiment selection

Nothing above B is claimed.
