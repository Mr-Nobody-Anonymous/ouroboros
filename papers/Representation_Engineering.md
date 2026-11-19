# Representation engineering and activation steering

- Zou et al., [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405), 2023.
- Turner et al., [Steering Language Models With Activation Addition](https://arxiv.org/abs/2308.10248), 2023.

Contrastive activation statistics can define a direction such as `v = mean(a_positive) - mean(a_negative)`, possibly normalized or estimated through PCA. A runtime intervention adds a scaled vector to selected activations. Its meaning depends on prompts, layer/token selection, controls and evaluation; a vector does not inherently guarantee truthfulness, safety or mathematical ability.

## Local status

`ArchitectureMutator.compute_steering_vector` implements finite, dimension-checked difference-in-means. `WeightSpaceProgrammer.contrastive_direction` separately exposes a measured direction for candidate tensor edits. Neither automatically installs activation hooks, fits all representation-reading methods or proves causal semantics.

`external/repeng` is a preserved upstream reference snapshot. Use its actual `ControlModel`/`ControlVector` workflow in a compatible isolated environment for runtime steering and measure behavioral side effects. Arithmetic tests of a vector are not an end-to-end representation-engineering benchmark. See [integration notes](../docs/INTEGRATIONS.md).
