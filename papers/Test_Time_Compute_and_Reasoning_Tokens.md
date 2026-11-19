# 🧠 Test-Time Compute Scaling & Self-Taught Reasoning (STaR & R1)

- **Seminal Publications**:
  - *STaR: Bootstrapping Reasoning With Reasoning* (Zelikman, Wu, Goodman, Stanford, 2022) — [arXiv:2203.14465](https://arxiv.org/abs/2203.14465)
  - *Large Language Monkeys: Scaling Inference Compute with Verifiers* (Brown et al., 2024) — [arXiv:2407.21787](https://arxiv.org/abs/2407.21787)
  - *DeepSeek-R1: Incentivizing Reasoning via RL & Test-Time Rollouts* (DeepSeek, 2025)
- **Primary Breakthrough**:
  - Rather than scaling model size (parameters) at pre-training time, models achieve massive performance jumps by scaling **inference-time computation** (generating multiple reasoning rollouts, self-verifying, and selecting the optimal path).
  - Successful reasoning traces are self-curated to fine-tune the model's own weights in subsequent generations (the STaR loop).

---

## 🧮 Mathematical Formulation

### 1. Best-of-$N$ Sampling with Outcome Verifier
Given input prompt $x$, sample $N$ independent chain-of-thought trajectories $\{y_1, y_2, \dots, y_N\} \sim \pi_\theta(\cdot|x)$:
$$y^* = \arg\max_{y_i} V_\phi(x, y_i)$$
where $V_\phi$ is a reward verifier model scoring logical validity.

### 2. The STaR Self-Evolution Objective
If candidate trajectory $y^*$ successfully derives the ground truth target $z$, the pair $(x, y^*)$ is added to the training set $\mathcal{D}_{STaR}$:
$$\mathcal{L}_{STaR}(\theta) = - \mathbb{E}_{(x, y^*) \sim \mathcal{D}_{STaR}} \left[ \sum_{t=1}^{|y^*|} \log \pi_\theta(y^*_t \mid x, y^*_{<t}) \right]$$
This iteratively internalizes successful test-time exploration into the model's permanent parameters.

---

## 💻 Integrated in Ouroboros
- Implemented in `ouroboros/core/test_time_compute.py`
- Exported in `ouroboros/core/__init__.py`
