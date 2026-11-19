# 📚 arXiv Literature Index: Self-Evolving & Weight-Editing AI

This directory compiles seminal peer-reviewed research papers and preprints on autonomous model self-modification, weight surgery, model arithmetic, and recursive self-play.

---

## 📑 Core Taxonomy of Self-Evolving Architectures

| Research Vector | Seminal Papers | arXiv Identifier | Core Mechanism |
| :--- | :--- | :--- | :--- |
| **Surgical Weight Editing** | *Locating and Editing Factual Associations in GPT (ROME)* | [arXiv:2202.05262](https://arxiv.org/abs/2202.05262) | Rank-one MLP projection updates $\Delta W = (v - W_0 k) \frac{k^T}{k^T k + \lambda}$ |
| **Mass Weight Surgery** | *Mass-Editing Memory in a Transformer (MEMIT)* | [arXiv:2210.07229](https://arxiv.org/abs/2210.07229) | Distributed residual rank-$k$ updates across multiple layers simultaneously |
| **Task Arithmetic** | *Editing Models with Task Vectors* | [arXiv:2212.04089](https://arxiv.org/abs/2212.04089) | Direct linear arithmetic on weight vectors: $\theta_{new} = \theta_{base} + \lambda \tau$ |
| **TIES & DARE Merging** | *Deep Model Merging without Training* | [arXiv:2311.03099](https://arxiv.org/abs/2311.03099) | Sparsification, sign-resolution, and rescaling of parameter deltas |
| **Self-Rewarding Evolution** | *Self-Rewarding Language Models* | [arXiv:2401.10020](https://arxiv.org/abs/2401.10020) | LLM acts as self-judge and self-trainer via iterative DPO cycles |
| **Self-Play Fine-Tuning** | *SPIN: Self-Play Fine-Tuning Converts Weak to Strong* | [arXiv:2401.01335](https://arxiv.org/abs/2401.01335) | Zero-human-data self-play minimax optimization |
| **Autonomous Code/Arch** | *The AI Scientist: Automated Discovery* | [arXiv:2408.06292](https://arxiv.org/abs/2408.06292) | Autonomous code mutation, experiment execution, and paper generation |

---

## 🔗 Implementation Mapping

Each paper's mathematical formulation maps directly into an Ouroboros module:
- **ROME / MEMIT**: `ouroboros/core/weight_editor.py` & `external/EasyEdit/`
- **Task Vectors & Weight Merging**: `ouroboros/core/task_arithmetic.py` & `external/mergekit/`
- **Internal Architecture Mutation**: `ouroboros/core/architecture_mutator.py`
- **Self-Rewarding & DPO Data**: `ouroboros/core/self_rewarding.py`
