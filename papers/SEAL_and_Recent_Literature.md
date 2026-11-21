# SEAL, continual adaptation, and research corrections

## Self-Adapting Language Models (SEAL)

Adam Zweiger, Jyothish Pari, Han Guo, Ekin Akyürek, Yoon Kim, Pulkit Agrawal. [arXiv:2506.10943](https://arxiv.org/abs/2506.10943); [official repository](https://github.com/Continual-Intelligence/SEAL). SEAL generates self-edits such as adaptation data/directives and optimizes them through an outer reinforcement-learning loop using downstream adaptation outcomes.

Ouroboros preserves the existing source snapshot in `external/SEAL`. Its new local SFT controller and verified proposal callbacks are useful infrastructure but do **not** reproduce SEAL's outer RL training. A regex hyperparameter edit alone is not SEAL. Source/snapshot provenance and integration levels are in [the inventory](../external/README.md).

## Recursive synthetic-data drift and replay

[AI models collapse when trained on recursively generated data](https://www.nature.com/articles/s41586-024-07566-y), Shumailov et al., Nature 2024, motivates keeping real-data support and measuring distributional degradation. It does not establish a universal “20–30% golden-anchor” prescription or a proof that a tiny memory bank eliminates collapse.

`AnchorMemoryBank` now implements configurable replay at the intended final-batch fraction with immutable copies and local RNG state. Callers must supply verified, licensed anchors. Duplicating a few examples is not a high-entropy foundational corpus and cannot guarantee no forgetting.

## Incorrect MoE-CL citation removed

The old index described **arXiv:2404.14818** as a self-evolving MoE/continual-learning paper with a zero-forgetting guarantee. The [actual arXiv page](https://arxiv.org/abs/2404.14818) is **Simplicial volume of manifolds fibering with connected structure group**, by Thorben Kastenholz, an unrelated topology paper. That AI attribution and guarantee were removed rather than propagated. No replacement MoE-CL implementation is claimed without a verified source.

## Harness and weight co-evolution

[SIA: Self Improving AI with Harness & Weight Updates](https://arxiv.org/abs/2605.27276), Hebbar et al., first submitted May 26, 2026 (v2 May 28), is the exact source replacing the previous vague 2025 homepage link. It co-updates harness and weights in selected domains. [Evo2](https://arxiv.org/abs/2610.13025), October 9, connects proposer/reflector harness changes with policy training. Neither algorithm is automatically implemented by the presence of a `CodeMutator` class.

For **11 actually recent sources in the August 19–November 19, 2026 window**, use [the dated review](RECENT_RESEARCH_2026.md) and [paper index](PAPERS_INDEX.md). Foundational 2024/2025 work is not mislabeled as “published in the last three months.”
