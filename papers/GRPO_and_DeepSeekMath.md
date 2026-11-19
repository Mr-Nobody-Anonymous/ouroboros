# DeepSeekMath and GRPO: reference, not a hidden optimizer

**Primary source:** Shao et al., [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300), 2024.

Group Relative Policy Optimization (GRPO) uses rewards within a sampled group to construct relative advantages, reducing reliance on a separately trained critic. It still requires a defined policy objective, actual token log probabilities, a trustworthy reward signal, KL/reference handling and substantial sampling/training compute. Sampling random “verifier scores” is not GRPO.

For group rewards r_i, an illustrative standardized advantage is `(r_i - mean(r)) / (std(r) + epsilon)`; this alone is **not the complete clipped/KL-regularized training objective**. Equal rewards, reward miscalibration, response-length effects and exploitable verifiers require care.

## Local status

Ouroboros preserves the existing TRL source snapshot and provides real verifier/judge interfaces, bounded rollout selection, data curation and local LoRA SFT. It does **not implement or automatically launch GRPO**. Use an isolated upstream TRL experiment for RL, record actual reward/compute evidence and wrap resulting checkpoints in independent evaluation/rollback. See [source inventory](../external/README.md) and [training guide](../docs/TRAINING.md).

This file was previously empty; it now explicitly documents both the research and the missing implementation instead of implying that cloning TRL completed reinforcement learning.
