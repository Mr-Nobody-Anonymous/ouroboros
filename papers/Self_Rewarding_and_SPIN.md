# Self-rewarding, preferences, DPO and self-play

Primary references:

- Rafailov et al., [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290), 2023.
- Yuan et al., [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020), 2024.
- Chen et al., [Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models](https://arxiv.org/abs/2401.01335), 2024 (SPIN).

DPO optimizes preferred versus rejected completions using their policy/reference log-probability ratios. A preference record requires a real prompt, chosen/rejected candidates and a defensible ranking signal. A learned model judging itself may share blind spots or reward-hack; stylistic words such as “therefore” are not factual correctness.

SPIN and self-rewarding methods have specific training procedures and data assumptions. A fixed riddle set is not a learned minimax opponent, and generating plausible preference JSON does not execute DPO or GRPO.

## Local status

`SelfRewardingEngine` now requires explicit generator/judge callbacks, validates finite scores, abstains on identical/tied candidates and preserves margins. It no longer fabricates “good” generic derivations or rewards reasoning keywords. `AdversarialCritic` supports independently verified challenge rounds and canonical answer checks; its built-in questions are test fixtures.

The default adaptation controller runs **SFT**, not preference/RL optimization. TRL is a preserved reference and optional dependency for separate upstream experiments. No self-rewarding/SPIN/DPO benchmark result is claimed. See [integration boundaries](../docs/INTEGRATIONS.md), [GRPO notes](GRPO_and_DeepSeekMath.md) and [security](../SECURITY.md).
