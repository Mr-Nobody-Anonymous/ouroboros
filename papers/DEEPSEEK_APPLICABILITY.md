# DeepSeek research: what was read, what was adopted, what was refused

This is a source-linked applicability review of the official **deepseek-ai** repositories and
their method sections, written to constrain what this repository may claim. It is **not** a
reproduction, and it is **not** exhaustive.

Machine-readable form (validated by tests, emitted by `ouroboros research deepseek`):
`ouroboros/research/applicability.py`.

## Corpus and method

- **19 pinned official repositories**: DeepSeek-LLM, DeepSeek-Coder, DeepSeek-MoE,
  DeepSeek-Math, DeepSeek-V2, DeepSeek-Coder-V2, DeepSeek-V3, DeepSeek-R1,
  DeepSeek-Prover-V1.5, DeepSeek-Prover-V2, ESFT, DeepSeek-Math-V2, DeepSeek-V3.2-Exp,
  Engram, DeepSeek-VL, DeepSeek-VL2, Janus, DeepSeek-OCR, DeepSeek-OCR-2.
- Pinned commits, blob ids, byte counts and SHA-256 for **62 documents** (including
  **13 method PDFs**) were resolved through the GitHub API before reading; the inventory is a
  review cache, not a distributable artifact. Moving `HEAD` was never substituted for a pin.
- Text was extracted from the cached PDFs and read section-by-section; several papers were
  additionally read from arXiv HTML. Extraction loses equations, figures and tables, some
  batched stdout was truncated, and no appendix was read end-to-end — so a mechanism may be
  described here more coarsely than in the paper.

**Where this review lives in the repository.** Each of the 19 repositories has its own
catalogue entry in [`research.json`](research.json) (with `deepseek_repo`, a review scope and
an explicit local status), a BibTeX record in [`references.bib`](references.bib), and a row in
[the research index](PAPERS_INDEX.md). A test
(`test_every_reviewed_deepseek_repository_is_present_in_the_repo_writing`) fails if a reviewed
repository lacks a catalogue/bibliography entry or if a listed title is missing from the index,
so this writing cannot silently drift from the machine-readable matrix.

Upstream **PDFs, extracted text and repository clones are not redistributed**: they were
reviewed from a temporary, hash-pinned cache and remain under the upstream repositories' own
licenses (code MIT per `LICENSE-CODE`, model/technical material under the DeepSeek model
license). Only titles, URLs, mechanism descriptions and local status are recorded here.

**Status vocabulary:** `implemented` (owned code, exercised by tests), `partial` (a bounded
subset, with listed material differences), `deferred` (understood, no implementation, no
claim), `not_applicable` (needs an architecture or data regime this loop does not have).

## Summary

| Status | Count | Meaning |
| --- | --- | --- |
| implemented | 3 | owned code with tests (data admission, distillation, training bookkeeping) |
| partial | 3 | real subset, material differences stated |
| deferred | 12 | no local implementation, listed for honesty |
| not_applicable | 4 | requires MoE/MLA/pretraining scale |

## Adopted (real, bounded)

### 1. Question-overlap data admission — DeepSeekMath corpus screening

DeepSeekMath describes iterative domain mining and **benchmark decontamination by n-gram
overlap** before training. The foundation adopts only the safety direction: a normalized
word n-gram (≤10 words, ≥3) shared between any visible and any hidden *question* refuses the
dataset before any model is loaded. It reports the witness pair and stops at the first
collision, keeping the check O(spans).

Adopted because leakage is a known way for a "gain" to be fake, and a strict pre-load refusal
is cheap. **Not** adopted: fastText classification, a 120B-token corpus, pretraining access,
semantic matching, answer-text screening. `semantic_decontamination: false`.

Code: `ouroboros/research/data_quality.py`, wired in `contracts.ResearchData.__init__`.

### 2. Verified distillation — DeepSeek-R1 rejection SFT

R1 filters teacher rollouts by *correctness* before distilling them. The `distill_sft` family
does exactly that shape, with three deliberate constraints: the teacher must be an explicitly
configured independent local model; it is sampled only on **training** prompts; and a
completion is accepted only if it matches the host reference exactly under the same
normalizer. Rejections are recorded per prompt.

Code: `worker.py` (`distill`), `evaluation.audit_rewards`.

### 3. Group-standardised reward audit — a diagnostic, explicitly not GRPO

DeepSeekMath's GRPO builds group-relative advantages inside a clipped, KL-regularised policy
objective. The foundation computes only the **standardisation** over bounded exact-answer
rollouts used as training-data filters, reports `uninformative_group` when the group is
constant, and never computes a policy gradient, ratio or KL term.

Naming discipline matters here: reward normalization is arithmetic, not reinforcement
learning, and no RL result is claimed.

### 4. Audited training bookkeeping — DeepSeek-LLM / V3 recipes

Cosine and constant schedules, gradient clipping, optimizer and model checkpoint cadence are
implemented as first-class, *measured* options: every step records loss, pre-clip gradient
norm, clip value, learning rate, example index, completion tokens, wall time and peak RSS,
and checkpoints export optimizer state plus CPU RNG as safetensors.

The papers' scale-specific hyperparameters are **not** transplanted to a small CPU target.

## Partial (honest subsets)

| Feature | Subset | Material difference |
| --- | --- | --- |
| DeepSeekMath GRPO | group reward standardisation | no objective, no log-probs, no KL, no reward model |
| Prover-V1.5 verified expert iteration | bounded rollouts + independent exact-answer check + fresh-worker re-evaluation | no Lean, no proof state, no tactic search; syntax ≠ correctness |
| DeepSeek-LLM scaling recipe | schedules/clipping/checkpoints measured locally | a 1M fixture cannot validate a scaling law |

## Deliberately refused

- **Math-V2 learned verifier and meta-verification.** A learned score can reward hallucinated
  flaws. The foundation keeps an exact-answer grader and never lets a model score its own
  work, so the failure mode has no configuration to slip into.
- **Engram as "memory".** The append-only SQLite ledger stores *experiments*; Engram is a
  trainable n-gram memory inside the forward pass. They are not the same thing and the docs
  say so wherever memory is mentioned.
- **ESFT as "LoRA on the right parts".** ESFT selects MoE experts from training gate
  statistics. Dense LoRA is not ESFT, and the interface reports the target's real
  architecture rather than implying expert selection.

## Deferred

R1 reasoning RL · Prover-V2 subgoal decomposition · V3.2-Exp sparse attention · Coder FIM and
repository mining · DeepSeek-VL hybrid encoders · VL2 tiling · Janus unified encoders · OCR and
OCR-2 optical compression. Vision work needs encoders, image data and visual evaluators that do
not exist here; each remains its own paper rather than a relabelled text feature.

## Not applicable to this target

V3 multi-token prediction and MoE routing · V2 MLA/KV compression · MoE fine-grained experts ·
Coder-V2 trillion-token continued pretraining. Each requires retraining from a different
initialisation, so post-hoc tweaks would be a different experiment wearing the paper's name.

## Per-source map

| Repository | Catalogue id | Local status |
| --- | --- | --- |
| DeepSeek-LLM | `deepseek-llm` | partial — schedule/clipping/step bookkeeping only |
| DeepSeek-Coder | `deepseek-coder` | reference only — no FIM, no repo mining |
| DeepSeek-Coder-V2 | `deepseek-coder-v2` | reference only — no continued pretraining |
| DeepSeek-MoE | `deepseek-moe` | not applicable — dense target |
| DeepSeek-Math | `deepseekmath` | partial — admission screen implemented, GRPO not |
| DeepSeek-Math-V2 | `deepseek-math-v2` | refused — no learned verifier |
| DeepSeek-V2 | `deepseek-v2` | not applicable — no MLA |
| DeepSeek-V3 | `deepseek-v3` | partial — audited step records only |
| DeepSeek-V3.2-Exp | `deepseek-v3-2-exp` | deferred — no sparse attention |
| DeepSeek-R1 | `deepseek-r1` | partial — verified distillation, no RL |
| DeepSeek-Prover-V1.5 | `deepseek-prover-v1-5` | partial shape — exact-answer, not Lean |
| DeepSeek-Prover-V2 | `deepseek-prover-v2` | deferred — no proof assistant |
| ESFT | `esft` | refused substitute — dense LoRA is not ESFT |
| Engram | `engram` | deferred — SQLite ledger is not model memory |
| DeepSeek-VL | `deepseek-vl` | deferred — no vision stack |
| DeepSeek-VL2 | `deepseek-vl2` | deferred — no vision stack |
| Janus | `janus` | deferred — no vision stack |
| DeepSeek-OCR | `deepseek-ocr` | deferred — no vision stack |
| DeepSeek-OCR-2 | `deepseek-ocr-2` | deferred — no vision stack |

## Non-claims

- No DeepSeek model weights, data or code are redistributed; nothing here is a reproduction.
- No source-reported benchmark number is a local result.
- No RL, formal verification, MoE, MLA, sparse attention, multimodal or Engram capability is
  implemented.
- Reading 62 documents is not a survey of DeepSeek research, and unread appendices may change
  the description of a mechanism.

## Related

- [Foundation architecture and status](../docs/RESEARCH_FOUNDATION.md)
- [Existing GRPO note](GRPO_and_DeepSeekMath.md) — the older reference-only entry, unchanged
- [Research index](PAPERS_INDEX.md)
