# Self-Instruct, synthetic curricula and feedback quality

- Wang et al., [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560), 2022.
- Cui et al., [UltraFeedback: Boosting Language Models with High-quality Feedback](https://arxiv.org/abs/2310.01377), 2023.

Self-Instruct bootstraps instructions/instances from seed tasks and filters the generated corpus. UltraFeedback studies richer AI feedback for model improvement. Neither establishes that a single model's unverified judgment is always reliable or that recursively generated data can replace distributional coverage indefinitely.

## Local status

The original `MetaFeedbackGenerator` returned a few repeated templates and was not a complete Self-Instruct implementation. It now exposes explicit generation callbacks and weak-domain selection without altering Python's global random state. The controller caps proposed examples, requires an independent verifier, rejects exact normalized validation overlap, deduplicates conflicting labels, records source/verification fields, and blends verified replay.

`external/self-instruct` remains the existing partial source snapshot; no original revision is invented. The built-in demo data are deliberately small local fixtures, not an autonomously generated research-quality corpus. Semantic deduplication, broad diversity auditing, contamination detection beyond normalized exact prompts, and expert/task-specific verification remain necessary for serious experiments.

See [data schema and callback example](../docs/TRAINING.md), [recent continual-learning evidence](RECENT_RESEARCH_2026.md), and [the source inventory](../external/README.md).
