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
| **Parameter-Efficient Tuning** | *LoRA & QLoRA: Quantized Low-Rank Adaptation* | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) | Intrinsic rank factorization $W = W_0 + \frac{\alpha}{r}(BA)$ |
| **Self-Instruct Bootstrapping** | *Self-Instruct: Aligning with Self-Generated Data* | [arXiv:2212.10560](https://arxiv.org/abs/2212.10560) | Autonomous synthetic task & pair generation |
| **Representation Engineering** | *Representation Engineering: Transparency & Steering* | [arXiv:2310.01405](https://arxiv.org/abs/2310.01405) | Dynamic activation steering vectors $x_l' = x_l + c \cdot v_{steer}$ |
| **Test-Time Compute Scaling (STaR)** | *STaR: Bootstrapping Reasoning & Inference Scaling* | [arXiv:2203.14465](https://arxiv.org/abs/2203.14465) | Best-of-N rollouts, verifiers & reasoning trace extraction |
| **Weight-Space Programming (WSP)** | *Sparks of Weight-Space Programming (MIT CSAIL)* | [arXiv/Takehi et al.](papers/Sparks_of_Weight_Space_Programming.md) | GPT-6 Astra editing 0.016% of Qwen weights; MATH-500 jumps 7.9% -> 48.1% |
| **Self-Adapting Models (SEAL)** | *Self-Adapting Language Models (MIT CSAIL)* | [arXiv:2506.10943](https://arxiv.org/abs/2506.10943) | Autonomous self-edits with RL outer-loop weight optimization |
| **Continual MoE Evolution** | *Self-Evolving LLMs via MoE-CL* | [arXiv:2404.14818](https://arxiv.org/abs/2404.14818) | Adversarial shared LoRA experts with task discriminators |
| **Entropy Collapse Protection** | *AI Models Collapse on Recursive Data (MAD)* | [Nature 631, 755](https://www.nature.com/articles/s41586-024-07566-y) | Proves necessity of Golden Anchor reference memory banks |
| **Co-Evolving Weights & Harness** | *SIA: Self-Improving AI with Weights & Scaffolds* | [HuggingFace 2025](https://huggingface.co) | Dual-knob co-evolution of agent harnesses and underlying weights |

---

## 🔗 Implementation Mapping

Each paper's mathematical formulation maps directly into an Ouroboros module and cloned framework:
- **Test-Time Compute & STaR Trajectories**: `ouroboros/core/test_time_compute.py`
- **Weight-Space Programming (Zero-Gradient Surgery)**: `ouroboros/core/weight_space_programmer.py`
- **ROME / MEMIT**: `ouroboros/core/weight_editor.py` & `external/EasyEdit/`
- **Task Vectors & Weight Merging**: `ouroboros/core/task_arithmetic.py` & `external/mergekit/`
- **Parameter-Efficient Adaptation**: `ouroboros/core/adapter_engine.py` & `external/peft/`
- **Self-Reward, DPO & GRPO**: `ouroboros/core/self_rewarding.py` & `external/trl/`
- **Representation Engineering & Steering**: `ouroboros/core/architecture_mutator.py` & `external/repeng/`
- **Self-Instruct Synthesis**: `ouroboros/core/meta_feedback.py` & `external/self-instruct/`
- **Autonomous Architecture Mutation**: `ouroboros/core/architecture_mutator.py`
