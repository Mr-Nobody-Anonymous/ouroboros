# Repository deep dive: audit and complementary upstream sources

Reviewed November 19, 2026. The repository name is already **ouroboros**; no remote rename is needed. Scope is the working branch based on `97181d3`. Historical chat messages are requests/claims to audit, not evidence that their described actions succeeded.

## Starting-state findings

The thirteen original tests passed, but mostly asserted shapes/strings. The default loop used random gradients unrelated to its examples, invented score boosts, saved only adapter metadata and displayed rollback without restoring parameters. A boolean simulation flag could not enable real training. The EasyEdit bridge returned success without invoking EasyEdit. The “Pareto” cleaner ranked mean scores without using file sizes and could delete unknown folders. Test-time exports could accept a wrong answer based only on a random high score. The sparse threshold mask could edit every tied parameter.

Seven source names existed, but only five were vendored trees. EasyEdit and mergekit were gitlinks without `.gitmodules`, so their empty directories looked available to bridges while not being reproducible from a clone. The old docs misattributed a topology arXiv ID to MoE-CL and overstated replay/no-forgetting guarantees. The old release's asset list was empty; no LFS pointer/payload or recoverable 600 MB cache bundle was found. See [migration and evidence](../docs/AUDIT_AND_MIGRATION.md).

## Why these sources, rather than cloning everything?

A useful stack needs complementary update surfaces **and** independent evaluation. More clones do not create a stronger model, resolve dependency conflicts or prove integration. We kept existing trees intact, repaired broken clone metadata, added five targeted sources, and recorded whether each is code, evidence, a paper-only placeholder or an actual callable adapter. We did not count repository popularity as proof of correctness or run unreviewed installers.

| Source / upstream | Mechanism and role | Actual local connection | Requirements / caveat |
| --- | --- | --- | --- |
| [PEFT](https://github.com/huggingface/peft) | Mature parameter-efficient adapter library | Real installed-package LoRA backend; existing vendored reference preserved | Local causal LM; optional CUDA/bitsandbytes for QLoRA; GPU path untested |
| [TRL](https://github.com/huggingface/trl) | DPO/GRPO and other post-training objectives | Existing source/reference and optional dependency; preference curation callbacks | Not executed by the default SFT loop; reward/reference/data design required |
| [EasyEdit](https://github.com/zjunlp/EasyEdit) | ROME/MEMIT/MEND/FT factual edits | Repaired pinned submodule; bridge calls real installed editor or fails | Method-specific config/covariance/model support; upstream may fetch data/use remote code |
| [mergekit](https://github.com/arcee-ai/mergekit) | Checkpoint merging, task vectors and data-flow recipes | Repaired pinned submodule; validated recipes and explicit subprocess execution | Matching architectures/tokenizers, model checkpoints, RAM/storage; LGPL license |
| [SEAL](https://github.com/Continual-Intelligence/SEAL) | Learned self-edits and outer RL | Existing source preserved; motivates verified data-proposal/controller separation | No SEAL outer RL reproduction; upstream GPU/data setup separate |
| [repeng](https://github.com/vgel/repeng) | Runtime activation control vectors | Existing source preserved; local vector primitives | Hooks are not automatically installed; behavioral effects require testing |
| [self-instruct](https://github.com/yizhongw/self-instruct) | Synthetic instruction bootstrapping/filtering | Existing partial snapshot preserved; real callback boundary replaces canned generator | No automatic large generated corpus; independent target verification necessary |
| [Original WSP logs](https://github.com/RikiyaT/lm_understand-logs) **new** | Original prompts, code, protocols, outcomes and artifacts | Cloned/read exp8 code/protocol; owned instrumentation, multi-tensor projections/controls and exact bundles | Mixed licenses; upstream scripts not executed, Astra/Qwen results not reproduced |
| [WSP-Scope](https://github.com/WSP-Scope/WSP-Scope-Project) **new** | Study of sparse edits, transfer and abstention | Pinned paper provenance | At this revision **PDF only**, no runnable code/license grant; not an engine |
| [WeightBench](https://github.com/Aayush-Marasini/WeightBench) **new** | Independent 42-task WSP evaluation and OS harness | Pinned external benchmark source | Requires separate sandbox/model/API setup; no local benchmark run claimed |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) **new** | Standardized independent LLM evaluation | Cloned/inspected CLI, owned plan-first offline external driver | Separately installed trusted tool/data/model required; actual benchmark not run |
| [Darwin Gödel Machine](https://github.com/jennyzzt/dgm) **new** | Empirical source-code evolution of coding agents | Pinned reference for proposal/archive workflows | Docker, benchmark data and APIs; not direct parametric self-rewriting |

Exact commits, snapshot hashes and licenses live in [sources.lock.json](../external/sources.lock.json). Existing vendored revisions remain unknown rather than guessed; content hashes make future local changes visible. There are **seven submodules plus five vendored snapshots**, not twelve automatically integrated trainer packages. New upstream trees were not patched secretly; adaptation is in owned modules.

## Implemented changes that close practical gaps

1. Minimal packaging/CLI/CI separate from optional heavy ML dependencies.
2. A backend protocol and actual gradient-trained CPU classifier for fast deterministic orchestration tests.
3. A local safetensors causal-LM PEFT LoRA backend with prompt/padding masking, gradient clipping/accumulation, deterministic evaluation and export/reload.
4. Data/source fingerprints, exact normalized split-leak checks, conflicting-label rejection and independently verified proposals/replay.
5. Measured original/latest gates, strict improvement/forgetting thresholds, real rollback after training/evaluation/export failure, and bounded generations/steps/cooperative time.
6. Actual tensor checkpoint files with atomic export, hashes and parent lineage, plus conservative actual-byte Pareto retention.
7. Exact sparse budgets, finite/shape checks, explicit copied structural candidates, TIES and arithmetic edge-case fixes. The 0.3 extension adds bounded numeric inspection, measured one-matrix trials with unconditional restoration, and accepted-only exact reversible data patches; this is owned tooling, not upstream benchmark reproduction.
8. No fake EasyEdit success, no random verifier/judge values, no acceptance of confident wrong STaR labels, no auto-execution of generated Python.
9. Original-source research notes with 33 references including 11 actually recent reviews, bibliography, limitations and machine-readable status.
10. Honest artifact inventory, narrow LFS attributes, non-destructive storage audit, deterministic core release builder and remote-publication separation.
11. Optional disjoint final-test data, blocked from proposals/promotion and evaluated once after selection. Curriculum callback/label failures now terminate with recorded stage evidence and owned-state rollback. Shared frozen local-LM inference is used by both PEFT and direct trials.

12. 0.4's real same-model finite-menu loop, explicit receipt-bearing model provisioning, packaged starter/installer and official-SDK stdio MCP. The pretrained model selected two nonzero edits; both plateaued/rejected/restored.
13. 0.5's separate [Research Lab](../docs/RESEARCH_LAB.md): frozen editor, model-authored JSON/numeric programs, multi-tensor transactions/instrumentation, paired controls/statistics, sealed reserved checks, independent sessions, exact full-state-bound bundles and explicit local-data/external-benchmark boundaries. The actual 135M model failed action authorship; a separately scripted two-tensor integration rejected at 0→0. No silent scripted substitution or paper gain.
14. A [feature-by-feature paper audit](../docs/PAPER_PARITY.md), source-informed implementation attribution and separate API/CLI/MCP evidence. General Python isolation, paper-specific task adapters and actual benchmark reproduction remain absent rather than mislabelled complete.

15. The opt-in [hypothesis-driven discovery profile](../docs/DISCOVERY_EXPERIMENTS.md) now couples prospective hypotheses to complete numerical intervention journals, actual iterative practice feedback, explicit sensitivity access and strict sealed development/reserved confirmation. It adds all eight requested tools together, fixed subsequent-edit retention, source-bound two-seed/eleven-arm controls and an explicit larger-model resource profile. Its hand-designed classifier positive control and real random-Llama gradient checks are mechanism evidence, not a pretrained capability claim; model studies/results are recorded separately.
16. OpenAI's [grade-school-math](https://github.com/openai/grade-school-math) was actually cloned at `3101c7d5072418e28b9008a6636bde82a006892c`, with README/MIT/train/test bytes reviewed. The owned adapter extracts 60 preselected disjoint GSM8K question/numeric-reference pairs across five roles, preserves original row hashes and notice, and executes no upstream code. The total 240-example evaluation also includes 180 synthetic diagnostics; it is not full GSM8K or semantic/pretraining decontamination.
17. [SmolLM2-360M-GGUF source](https://github.com/Dadhichi-Sarker-Shayon/SmolLM2-360M-GGUF) was actually cloned at `a3f5ab2509573383064fd986bfff2d912dc4ba39` and reviewed for conversion/size/license provenance. Its metadata refer to base, not Instruct; no payload or upstream converter was run. The separate CI provisions official 360M Instruct with its own resolved revision/byte receipt before inference. [Source review hashes and limits](../docs/evidence/discovery-sources.json). These development clones are not silently treated as extra preserved submodules.

## Still not solved

A learned autonomous experiment proposer, a complete SEAL/EvoTIDE/DPO/GRPO outer training loop, robust semantic contamination detection, hard OS/VRAM quotas, arbitrary architecture/quantization support, broad independently benchmarked capability gains, and a universal self-judge are **not implemented merely by adding these tools**. Statistical uncertainty, held-out final tests, multiple seeds, locality/safety evaluation and provenance of the original model still matter.

See [architecture](../docs/SELF_EVOLVING_ARCHITECTURE.md), [testing](../docs/TESTING.md), [integration setup](../docs/INTEGRATIONS.md) and [recent primary-source review](../papers/RECENT_RESEARCH_2026.md).
