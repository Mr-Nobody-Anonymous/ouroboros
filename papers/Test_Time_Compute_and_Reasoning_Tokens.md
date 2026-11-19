# Test-time sampling, verification and correctness-filtered training exports

- Zelikman et al., [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465), 2022.
- Brown et al., [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/abs/2407.21787), 2024.

Best-of-N samples candidates and chooses according to an outcome verifier; majority voting uses agreement, which is not the same as correctness. STaR iteratively turns verified successful reasoning into training data. More sampling consumes compute and can amplify a bad verifier; comparisons must account for token budgets and test contamination.

## Local status

`TestTimeComputeEngine` now requires real generator/verifier callbacks instead of random mock scores. It supports bounded candidate budgets, finite score validation, best/second-best margin and majority voting. Its export helper only accepts an exact canonical ground-truth final answer: a confident but wrong candidate cannot become a training label merely because its judge score exceeds 0.85.

This is **partial correctness-filtered curation**, not a complete STaR rationalization/retraining algorithm or a reproduction of a frontier reasoning model. The default SFT controller does not silently run Best-of-N or train on held-out evaluation traces. Use task-specific answer extraction and independent verifiers for noncanonical math/code outputs. See [training](../docs/TRAINING.md) and [test evidence](../docs/TESTING.md).
