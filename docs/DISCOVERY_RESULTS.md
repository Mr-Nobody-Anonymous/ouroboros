# Closed-loop discovery: reviewed results, including failures

**Bottom line:** the complete 360M studies produced **zero model-authored numerical
edits and zero accepted candidates**. A real 135M study is still running; its first seven
completed arms also contain no accepted autonomous edit. The tooling works mechanically,
but **useful autonomous pretrained self-editing and learned future self-improvement have
not been demonstrated**. This is a negative scientific result, not a failed integrity check.

[Protocol and reproduction](DISCOVERY_EXPERIMENTS.md) ·
[Complete 360M compact evidence](evidence/discovery-360m-reviewed.json) ·
[Explicitly partial 135M snapshot](evidence/discovery-135m-partial.json)

## Which execution was actually reviewed?

| Item | Verified scope |
| --- | --- |
| Original source-bound runtime | `cb4c0df1008f208016a8bfa941c999bd96bcfc8b`; 63 Python source bindings agree across the local and remote plans |
| Original 360M execution | [Push 44715379570](https://github.com/Mr-Nobody-Anonymous/ouroboros/actions/runs/44715379570): all nine jobs passed, including both eleven-arm discovery jobs |
| Checkpoint | Official `HuggingFaceTB/SmolLM2-360M-Instruct`, **361,821,120 parameters**, immutable resolved revision `a10cc1512eabd3dde888204e902eca88bddb4951`; receipt binds all eight downloaded files |
| Completed scientific scope | All eleven arms at each fixed seed, 17 and 42; **22/22 remote arms**, not selected favorable notices |
| Read-only archive review | [44725072525](https://github.com/Mr-Nobody-Anonymous/ouroboros/actions/runs/44725072525), exporter `b0cf0a7b35a596f1b2b44d0b4997d1ff2b81fe4b`; no model rerun, new evaluation, candidate selection or gate change |
| Original ZIPs | Seed 17: artifact **14623630301**, **838,978 bytes**; seed 42: **14621214992**, **812,097 bytes**. Their complete SHA256s are in the evidence receipt |
| Local study | Receipt-verified, Q4_1-derived float32 135M checkpoint, **134,515,008 parameters**; first **7/22 planned arms** reviewed, process continues under its original plan |

GitHub downloaded the **existing** archives, verified their inventory sizes/digests, safely
extracted JSON and verified original report, journal, event and retention-artifact hashes.
The full aggregate/numerical exports were then recovered locally from **three Checks API
parts per seed**, concatenated, strictly base64-decoded, decompressed and SHA256-verified:

| Seed | Decoded payload bytes | Uncompressed export SHA256 |
| --- | ---: | --- |
| 17 | 514,000 | `b335c260c5002ed2d5a66a13f476122e81cf5d952e67c1ae6b9459c4d3484d4b` |
| 42 | 512,692 | `12035786865ddbb53063ef45bd51b758d9f5e4bf5a2f39d4c2714fc6301f6de5` |

The sandbox **did not download the original ZIPs locally**: its Azure redirect failed.
Oversized original Actions notices were truncated to 4,096 characters and capped at ten
retrievable notices per seed; they were **not** used as complete evidence. The successful
hosted archive verification plus lossless, hash-verified Checks transport resolves that
access problem without pretending the first download worked.

The complete exported arrays were independently checked locally: every attempted edit and
rollback, both available fixed-follow-up ledgers, all stored before/after values and their
recomputed delta norms. Full-tensor replay and model/source restoration are **hash-bound
assertions from the original runner**, not an independent rerun of the model or a
reconstruction of unedited tensor entries. Raw hidden prompts/answers are not in the compact
receipts or the Checks export. Identical aggregates are explicitly deduplicated by local
JSON Pointer references; they are not additional measurements.

## 360M: syntax, semantics and numerical edits are different outcomes

Practice baseline was **7/32 exact**, all seven from copying (**7/8**); the other three
categories scored **0/8** each. Problem-mean reference+EOS NLL was **2.0430097012**. These
are strict completion scores, not a claim that every non-exact answer was factually wrong.
Both greedy baselines are identical; repeated answers/seeds are not independent benchmark
replications.

The table below reports **host-valid commands / actual editor replies**. Every model arm
made **zero valid nonempty edits**, so none measured the numerical usefulness of its own
parameter proposal.

| Model-authored arm | Seed 17 | Seed 42 |
| --- | --- | --- |
| `unconstrained` | **0/4**, invalid JSON, budget exhausted | **0/4**, invalid JSON, budget exhausted |
| `schema` | **3/5**, read-only `notes`, host-stalled | **0/4**, invalid hypotheses, budget exhausted |
| `model_directed_gradient` | **3/5**, read-only `notes`, host-stalled | **0/4**, invalid hypotheses, budget exhausted |
| `schema_no_feedback` | **3/5**, read-only `notes`, host-stalled | **0/4**, invalid hypotheses, budget exhausted |
| `schema_full_context` | **0/4**, one truncated reply and invalid hypotheses | **1/1**, submitted an empty candidate, withdrawn without a gate look |
| `schema_sparse_ops` | **0/4**, invalid hypotheses/saved name | **0/4**, invalid hypotheses, budget exhausted |
| `schema_staged` | **0/4**, invalid hypotheses, budget exhausted | **0/4**, invalid hypotheses, budget exhausted |
| `schema_staged_gradient` | **0/4**, invalid hypotheses, budget exhausted | **0/4**, invalid hypotheses, budget exhausted |

Across these arms there were **64 editor replies, 55 parsed commands and 10 host-valid
commands**, but **zero accepted hypotheses, gradient requests or numerical edit attempts**.
Most parsed hypotheses predicted zero or a wrong-sign effect, which the unchanged host
validator refused. Structural grammar does not enforce scientific/semantic validity.
Repeated read-only calls ended via the host's no-progress rule; `stalled` is **not** a
model-authored decision to abstain. The expanded-context seed-42 empty `submit` also does
not constitute a useful edit.

No remote arm opened development or reserved evaluation. All selected final candidates
therefore remained base: **7/64 exact**, problem-mean NLL **1.5605650579**, unchanged. No
failed gate, final diagnostic or loss decrease was relabelled as acceptance.

## 360M: explicit HOST controls really changed weights, but did not pass

The host control used `model.layers.31.post_attention_layernorm.weight`, **256 entries**
per proposal, at the two predefined relative doses. Both seeds produced the same host
programs/greedy observations; these are not two independent discoveries.

| HOST trial | Actual delta norm | Practice exact | Practice NLL | Outcome |
| --- | ---: | --- | --- | --- |
| First, relative dose 0.01 | 0.1006616320 | **7/32→6/32** | **2.043010→2.253606** | Harmful, identified and rolled back |
| Second, relative dose 0.03 | 0.3019848780 | **7/32→6/32** | **2.043010→3.700698** | Harmful, identified and rolled back |

Recorded partial first-order NLL predictions were **−0.106741** and **−0.320223**; the
finite interventions instead increased NLL by **+0.210596** and **+1.657688**. This is a
measured failure of these fixed doses, not proof of a particular nonlinear mechanism or
of semantic/causal weight understanding. The unit finite-difference gradient check is
separate from a successful pretrained positive control.

Matched-random trials used the **same actual tensor and support**, with relative actual-norm
discrepancies no larger than **2.26e−7**:

| Seed | First random trial: exact; NLL delta | Second random trial: exact; NLL delta |
| --- | --- | --- |
| 17 | 7/32→7/32; **−0.00002248** | 7/32→7/32; **+0.00956179** |
| 42 | 7/32→7/32; **−0.01122661** | 7/32→5/32; **+0.12200066** |

Both random policies reversed the first trial as specified and the harmful second trial,
then submitted an empty candidate and withdrew. A practice-only NLL decrease is not useful
accuracy improvement. Their support is host-gradient-informed, not a wholly gradient-free
random null. All control interventions and reversals remain in the numerical ledger.

### Preserve the uncomfortable final/retention diagnostics too

The terminal diagnostic is the **last valid nonempty proposal**, even when harmful and
already rolled back. It is not the chosen model or the best earlier dose. The last HOST
proposal scored **7/64→12/64** on final exact accuracy, but lost arithmetic **1/16→0/16**,
worsened NLL **1.560565→3.355319**, and had **p=0.0625**, above the unchanged **0.05/44**
threshold. It cannot rescue a withdrawn candidate or become a newly selected dose.

On retention, that same diagnostic changed base **8/64→14/64**; the fixed subsequent
index-0 `+0.001` perturbation left **14/64**, but NLL remained much worse than base.
The no-edit probe retained **8/64** while a tiny copying-domain NLL regression remained
visible. The seed-42 random diagnostic failed the category-retention floor after the
follow-up (copying **8/16 base→6/16 combined**, despite offsetting arithmetic answers).
That copying loss was already present before the follow-up; it is not attributed to the
subsequent edit. **Useful-edit retention is null for every arm**, because no useful edit
was accepted. None of these
post-search probes is extended training or evidence of a learned editing policy.

## 135M: partial, not silently completed

The [frozen partial receipt](evidence/discovery-135m-partial.json) covers seed 17's first seven
completed arms. The original two-seed, eleven-arm process continues; later live progress
must not be inferred from this snapshot.

- `no_edit`: practice **0/32**, NLL **3.9946031943**; no intervention.
- **HOST** first dose: **0/32→3/32**, NLL **3.994603→3.675749**. Second dose:
  **3/32→0/32**, NLL **3.675749→6.954666**, identified as harmful and rolled back.
  Development **0/32→1/32, p=0.5** failed the original gate; **reserved unopened**.
  This is practice controllability, not validated pretrained usefulness or autonomy.
- Matched-random control: two actual 256-entry proposals, matching support/norms, no accepted
  edit. Neither losses nor the final diagnostic override its failed selection gate.
- `unconstrained` and `schema`: four replies each, **zero host-valid commands/edits**.
  The schema arm had three truncated replies and an invalid saved name.
- `model_directed_gradient`: six replies, **two valid commands** (`inspect_gradients` and
  `save`), **zero numerical edits**. It really queried practice sensitivity; unused-tool
  explanations alone cannot describe this trajectory. Four replies were truncated; the
  final generated reply hit a cooperative search budget before command execution.
- `schema_no_feedback`: four replies, **zero valid commands/edits**; three truncated
  replies and the same invalid saved-name refusal. Without an edit there was no edited
  feedback treatment to compare.

No source, model, data, seed list or threshold was changed under the running study, and it
was not restarted after inspecting its outcomes. A failed/incomplete arm will remain in
the evidence rather than being overwritten.

## What can and cannot be concluded

- **Established here:** the broader interface, actual feedback loop and numerical audit
  mechanism execute; explicit HOST interventions can change practice behavior and be
  reversed; these particular pretrained model policies did not produce useful edits.
- **Not isolated:** the reason for autonomous failure. The 360M access arms never queried
  gradients, and no 360M model arm reached edited feedback. Feedback-off/access comparisons
  therefore do not estimate the value of feedback or gradients. Small models, prompts,
  semantic constraints and fixed budgets remain alternative explanations.
- **Positive-control limit:** no pretrained HOST control passed the complete useful-edit
  gates. Do not attribute failure solely to model reasoning or parameter selection when
  the full successful mechanism has not been positively controlled on these checkpoints.
- **Statistical limit:** small, partly template-related cohorts; strict, low-powered
  44-submission correction; fixed condition order and repeated greedy answers. Gates were
  not weakened to obtain a success. The 240 examples include a fixed 60-item GSM8K
  selection, **not** a full benchmark or semantic/pretraining-decontaminated evaluation.
- **Scale/source limit:** 135M quantization-derived versus original 360M differs in source
  and precision as well as size. This is not a pure size-effect estimate or paper parity.
- **Meta-learning limit:** search-time feedback is not training an editing-policy checkpoint.
  No learned improvement of future self-improvement, durable generalization or extended-
  training retention has been demonstrated. A new study would require its own prior plan,
  unseen editing/evaluation tasks and unchanged independent success criteria.

All 22 remote arms asserted full resident/source restoration. The PR remains unmerged;
these are development-branch results, not a release or a claim that adding APIs has solved
the scientific problem.
