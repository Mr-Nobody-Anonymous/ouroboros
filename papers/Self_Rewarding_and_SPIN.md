# Autonomous Self-Rewarding & Self-Play Fine-Tuning (SPIN)

- **Papers**:
  - *Self-Rewarding Language Models* (Yuan et al., Meta & NYU, 2024) — [arXiv:2401.10020](https://arxiv.org/abs/2401.10020)
  - *SPIN: Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models* (Chen et al., UCLA, 2024) — [arXiv:2401.01335](https://arxiv.org/abs/2401.01335)
- **Primary Breakthrough**:
  - LLMs possess higher discriminative capability (judging) than pure generation capability.
  - The model can iteratively evaluate its own candidate responses, assign rewards via LLM-as-a-Judge, and fine-tune its own weights using Direct Preference Optimization (DPO).

---

## 🧮 Mathematical Formulation

### 1. Direct Preference Optimization (DPO) Loss
For preferred response $y_w$ and rejected response $y_l$ scored by the model itself:
$$\mathcal{L}_{DPO}(\theta; \theta_{ref}) = - \mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

### 2. Self-Play Minimax Formulation (SPIN)
Main Player $p_\theta$ plays against Opponent $p_{\theta_{t-1}}$:
$$\min_\theta \max_{p'} \mathbb{E}_{y \sim p_{data}} [\log p_\theta(y|x)] - \mathbb{E}_{y' \sim p'} [\log p_\theta(y'|x)] - \text{KL}(p' \| p_{\theta_{t-1}})$$
This moves model distribution closer to the target frontier without requiring new human annotations.

---

## 💻 Integrated in Ouroboros
- Implemented in `ouroboros/core/self_rewarding.py`
