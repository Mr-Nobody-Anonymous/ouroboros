# Task arithmetic, TIES, DARE and interpolation

These are distinct primary sources:

- Ilharco et al., [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089), 2022.
- Yadav et al., [TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708), 2023.
- Yu et al., [Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch](https://arxiv.org/abs/2311.03099), 2023 (DARE).

A task vector is `tau = theta_finetuned - theta_base`, applied as `theta_new = theta_base + lambda*tau`. Subtracting a vector may reduce a measured behavior but is **not certified knowledge removal**, robust unlearning, or a guarantee of removing toxicity/hallucinations.

TIES trims small-magnitude entries, elects a sign from aggregate signed magnitude, then averages compatible nonzero entries. DARE samples a keep mask with probability `1-p` and rescales retained entries by `1/(1-p)`. Dropping 100% is undefined and must be rejected. SLERP interpolates directions; parallel, zero and antipodal edge cases require explicit handling.

## Local status

`TaskArithmeticEngine` has finite/shape checks, seeded caller-supplied RNG support for DARE, exact trimming for tied magnitudes, TIES sign/disjoint merging and SLERP fallbacks. These are **reference tensor implementations**, not evidence of equivalent capability gains on arbitrary model checkpoints. Shape-compatible parameters do not ensure tokenizer, training history or semantic compatibility.

`MergeKitBridge` writes reviewed whole-model SLERP or DARE-TIES recipes and can explicitly call an installed `mergekit-yaml` process. Recipe generation does not merge anything. The controller does not blindly merge every new adapter before scoring it. See [integration boundaries](../docs/INTEGRATIONS.md).
