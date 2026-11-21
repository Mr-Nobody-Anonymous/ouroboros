# Sparks of Weight-Space Programming: verified source and implementation boundary

**Source:** Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab, MIT CSAIL. November 2, 2026. [Primary research blog](https://rikiyat.github.io/blog/weight-space-programming/), [original logs and harness](https://github.com/RikiyaT/lm_understand-logs). Consulted November 19, including the math experiment and limitations.

## What matches the remembered result?

The source describes GPT-6 Astra editing **Qwen3.5-0.8B**, not a Harvard study. It is a **blog/research report with no formal DOI/publication listed on the page**, not a verified arXiv article. The authors report:

| Metric | Base | 50-forward-pass edit |
| --- | ---: | ---: |
| MATH-500 accuracy | 7.9% | 48.1% |
| GSM8K accuracy | 13.8% | 67.1% |
| IFEval | 21.0% | 62.3% |

These are source-reported numbers for a specific setup, not measurements from this repository. The MATH-500 change is **+40.2 percentage points**, approximately 6.09 times baseline accuracy. Percent and percentage points are not interchangeable.

## What was actually edited?

The post describes a Python environment without gradient training, a limited forward-pass budget, contrastive productive/repetitive activation analysis, layer-isolation and strength experiments, and a norm-matched random-direction control. The final patch changes selected MLP output columns at layers 15 and 19: 64 columns each, **131,072 entries**, about 0.016% of total parameters. It attributes the benefit to reducing pathological repeated reasoning that exhausted the token budget.

That is **not equivalent** to choosing the largest singular vector of an arbitrary matrix or thresholding the largest deltas across the whole model. A singular direction shows algebraic structure, not causal proof of a repetitive attractor. The original code's universal SVD diagnosis and fixed-percentage pipeline stage overstated what had been implemented.

The post's Gemma 3 270M follow-up did not establish a comparable improvement. The authors distinguish repairing an obstacle to an existing skill from adding an absent skill. The result is neither a guarantee of general mathematical learning without gradients nor evidence that any small model can autonomously improve itself.

## What Ouroboros now provides

- Pinned `external/weight-space-logs` source with original protocols/evidence; no modified or fabricated experiment logs.
- An explicitly labeled `dominant_weight_direction` utility and deprecated misleading diagnostic name.
- `contrastive_direction` from caller-supplied failing/successful activation matrices.
- Shape/finite/zero-vector checks and bounded directional ablation.
- Exact maximum edited-entry budgets, including ties, zero and very small tensors.
- Bounded read-only safetensors metadata/prefix inspection without loading a model; numeric values do not reveal arbitrary semantics.
- Explicit measured single-tensor trials: paired baseline/candidate inference, actual rounded entry/norm/payload-byte limits and improvement/retention gates.
- Unconditional live-target restoration, even on acceptance, and accepted-only data-only sparse patches with exact explicit application/reversal against complete before/after target-tensor hashes.
- Independent optional final-test evaluation after SFT selection, excluded from generated curricula/promotion, rather than invented benchmark boosts.

The separate **0.5 [Research Lab](../docs/RESEARCH_LAB.md)** adds actual hidden/MLP/logprob/opt-in-attention reads, measured contrast directions, writer-column ranking, atomic multi-layer projections, signed strength/dose sweeps, saved alternatives, no-op/norm-matched random/layer-isolation controls, paired problem-seeded evaluation and boxed-math grading. It also provides frozen same-model sessions, sealed reserved validation and exact full-state-bound bundles, inspired jointly by the later self-rewriting paper. The old finite-menu/single-matrix path remains separate.

See the [weight-edit guide](../docs/WEIGHT_EDITING.md) for its runnable legacy CLI and limits. Its `edited_fraction` bounds the **selected matrix**; the new lab's global fraction bounds all model parameters. Neither setting establishes Astra's actual edit. The 2×2 edit demo and random-LM tests establish mechanics, not meaningful pretrained improvement. Real 135M research attempts produced zero valid actions; the separate scripted two-layer integration changed 1,152 entries, 0→0/rejected/restored. [Measured evidence](../docs/TESTING.md).

These are reusable **experimental tools**, not a reproduction of Astra's discovery process or exact released patch. Core SFT does not run this surgery automatically, and its adapter-only snapshots cannot capture arbitrary base edits. Tiny changes do not guarantee locality, no forgetting or unrelated capability preservation. Legacy patch hashes bind only the target tensor; research bundles additionally bind the full model state. Neither authenticates the author or legal rights. No persistent arbitrary-Python sandbox, dedicated game/diffusion/Go/vision experiment or real standard benchmark is established; see the [complete gap checklist](../docs/PAPER_PARITY.md).

## Related recent work

- [WSP-Scope, 2611.09382](https://arxiv.org/abs/2611.09382): sparse interventions, uneven generalization, six of sixteen trajectories abstain.
- [WeightBench, 2611.18553](https://arxiv.org/abs/2611.18553): independent task-based evaluation including impossible edits and failures.
- [Reading audit, 2611.08070](https://arxiv.org/abs/2611.08070): dataset priors can masquerade as semantic weight understanding.
- [Replication, 2611.21449](https://arxiv.org/abs/2611.21449): small-network writing replicates with narrower interpretation.
- [Self-rewriting, 2611.21885v2](https://arxiv.org/abs/2611.21885v2): a 27B model reportedly edits itself; smaller variants may fail.

Read [the recent review](RECENT_RESEARCH_2026.md) for scope/limitations and [SECURITY.md](../SECURITY.md) before executing any generated intervention.
