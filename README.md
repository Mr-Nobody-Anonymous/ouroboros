# 🐍 Ouroboros: Autonomous Recursive Self-Fine-Tuning Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Research: arXiv Literature](https://img.shields.io/badge/Research-arXiv%20Verified-brightgreen)](papers/PAPERS_INDEX.md)
[![Submodules: EasyEdit & MergeKit](https://img.shields.io/badge/Submodules-EasyEdit%20%7C%20MergeKit-orange)](external/)

**Ouroboros** is an autonomous framework engineered for Large Language Models (LLMs) to inspect, critique, and recursively fine-tune their own mathematical weight matrices, biases, internal representations, and architectures. 

By combining **Parameter-Efficient Fine-Tuning (PEFT/LoRA)**, **Surgical Model Editing (ROME/MEMIT)**, **Task Vector Arithmetic & DARE/TIES Merging (MergeKit)**, **Internal Architecture Mutation**, and **Self-Rewarding DPO Loops**, Ouroboros enables AI systems to continuously evolve their cognitive frontier across successive generations.

---

## 🏛️ Comprehensive Architecture

```
                                  ┌──────────────────────────────┐
                                  │   Generation N Baseline      │
                                  └──────────────┬───────────────┘
                                                 │
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │   Autonomous Self-Diagnosis  │
                                  │   • Identify weak domains    │
                                  │   • Analyze reasoning flaws  │
                                  └──────────────┬───────────────┘
                                                 │
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │   Self-Rewarding Synthesis   │
                                  │   • LLM-as-a-Judge scoring   │
                                  │   • DPO (Chosen vs Rejected) │
                                  └──────────────┬───────────────┘
                                                 │
                                                 ▼
             ┌───────────────────────────────────┴───────────────────────────────────┐
             │                                                                       │
             ▼                                                                       ▼
┌──────────────────────────────┐                                    ┌──────────────────────────────┐
│  Direct Weight Surgery &     │                                    │   Architecture Mutation &    │
│  Parameter Adaptation        │                                    │   Model Vector Arithmetic    │
│  • LoRA ΔW = (α/r)·(B·A)     │                                    │   • Task Vectors θ = θ_0 + λτ│
│  • ROME / MEMIT rank updates │                                    │   • DARE & TIES Weight Merge │
│  • EasyEdit Integration      │                                    │   • Layer Stacking & Pruning │
└────────────┬─────────────────┘                                    └──────────────┬───────────────┘
             │                                                                     │
             └───────────────────────────────────┬─────────────────────────────────┘
                                                 │
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │  Post-Tuning Verification    │
                                  │  • Multi-domain benchmarks   │
                                  │  • Forgetting detection      │
                                  └──────────────┬───────────────┘
                                                 │
                                    [Pass]       ▼       [Degrade]
                                  ┌──────────────────────────────┐
                                  │  Advance to Gen N+1          │
                                  │  (Or rollback checkpoint)    │
                                  └──────────────────────────────┘
```

---

## 📚 arXiv Research Papers & Literature Compilation

Located in the [`papers/`](papers/) directory:

1. **[ROME & MEMIT](papers/ROME_and_MEMIT.md)**:
   * *Locating and Editing Factual Associations in GPT* ([arXiv:2202.05262](https://arxiv.org/abs/2202.05262))
   * *Mass-Editing Memory in a Transformer* ([arXiv:2210.07229](https://arxiv.org/abs/2210.07229))
2. **[Task Arithmetic & DARE](papers/Task_Arithmetic_and_Weight_Vectors.md)**:
   * *Editing Models with Task Arithmetic* ([arXiv:2212.04089](https://arxiv.org/abs/2212.04089))
   * *Deep Model Merging without Training: TIES & DARE* ([arXiv:2311.03099](https://arxiv.org/abs/2311.03099))
3. **[Self-Rewarding Evolution](papers/Self_Rewarding_and_SPIN.md)**:
   * *Self-Rewarding Language Models* ([arXiv:2401.10020](https://arxiv.org/abs/2401.10020))
   * *SPIN: Self-Play Fine-Tuning Converts Weak to Strong* ([arXiv:2401.01335](https://arxiv.org/abs/2401.01335))
4. **[Automated AI Discovery](papers/Automated_Self_Evolution_AI_Scientist.md)**:
   * *The AI Scientist: Automated Discovery* ([arXiv:2408.06292](https://arxiv.org/abs/2408.06292))
   * *Darwinian Evolution of LLM Topologies* (Sakana AI, 2024)
5. **[PEFT, Self-Instruct & Representation Engineering](papers/PAPERS_INDEX.md)**:
   * *LoRA & QLoRA* ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685), [arXiv:2305.14314](https://arxiv.org/abs/2305.14314))
   * *Self-Instruct: Aligning Language Models with Self-Generated Data* ([arXiv:2212.10560](https://arxiv.org/abs/2212.10560))
   * *Representation Engineering: Steering Vectors* ([arXiv:2310.01405](https://arxiv.org/abs/2310.01405))
   * *DeepSeekMath & GRPO: Critic-Free Policy Optimization* ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300))

---

## 🔬 Comparative Analysis of Cloned Repositories

| Framework | Core Mechanism | Mathematical Formulation | Role in Ouroboros |
| :--- | :--- | :--- | :--- |
| **[EasyEdit](external/EasyEdit)** | Localized MLP memory editing | Closed-form $\Delta W = (v - W_0 k)\frac{k^T C^{-1}}{k^T C^{-1} k}$ | Surgical factual corrections without retraining |
| **[mergekit](external/mergekit)** | Weight tensor merging & arithmetic | SLERP, TIES sign election, DARE random mask & rescale | Combining specialized model vectors & frankenmerging |
| **[SEAL](external/SEAL)** | Self-adapting language models (MIT) | Self-edits via SFT & outer RL loop | Autonomous self-adaptation & optimization directive generation |
| **[peft](external/peft)** | Low-rank decomposition (LoRA/QLoRA) | $\Delta W = \frac{\alpha}{r}(B \cdot A)$ | Generation-to-generation parameter fine-tuning |
| **[trl](external/trl)** | Reinforcement learning & alignment | DPO loss & GRPO relative group advantage | Self-rewarding policy optimization |
| **[repeng](external/repeng)** | Dynamic activation steering | $x_l' = x_l + c \cdot v_{PCA}$ | Non-destructive runtime cognitive steering |
| **[self-instruct](external/self-instruct)** | Autonomous task bootstrapping | ROUGE-L diversity filtering & task synthesis | Generating self-correction curricula |

---

## 📂 Repository Layout

```plaintext
ouroboros-tuner/
├── ouroboros/
│   ├── core/
│   │   ├── weight_editor.py        # Direct tensor surgery, SVD rank updates, LoRA deltas & clamping
│   │   ├── adapter_engine.py       # PEFT/LoRA parameter tuning & gradient steps
│   │   ├── meta_feedback.py        # Autonomous error diagnosis & critique synthesis
│   │   ├── evaluator.py            # Multi-domain benchmark engine & anti-catastrophic forgetting
│   │   ├── task_arithmetic.py      # Task vectors (θ_0 + λτ), unlearning (-βτ), DARE, and SLERP
│   │   ├── architecture_mutator.py # Layer stacking, attention head scaling, activation steering & pruning
│   │   ├── self_rewarding.py       # Self-rewarding rubric, LLM-as-a-judge, DPO dataset generation
│   │   ├── anchor_memory.py        # Golden Anchor Memory Replay preventing entropy collapse (MAD)
│   │   ├── adversarial_critic.py   # Dual-agent Generator vs. Skeptic adversarial stress testing
│   │   ├── code_mutator.py         # Autonomous loss function & hyperparameter code mutation
│   │   └── pareto_pruner.py        # Automated Pareto-optimal checkpoint pruning to preserve disk space
│   ├── pipeline/
│   │   └── self_loop.py            # Recursive evolution loop orchestrator
│   ├── integrations/
│   │   ├── easyedit_bridge.py      # Interface to EasyEdit (ROME, MEMIT, MEND)
│   │   └── mergekit_bridge.py      # Interface to MergeKit (TIES, DARE, SLERP recipes)
│   ├── config.py                   # Configuration dataclasses
│   └── cli.py                      # Command-line entrypoint
├── external/
│   ├── EasyEdit/                   # Cloned knowledge editing repository
│   ├── mergekit/                   # Cloned model weight merging & arithmetic repository
│   ├── SEAL/                       # Cloned MIT CSAIL self-adapting language model framework
│   ├── peft/                       # Cloned Hugging Face parameter-efficient fine-tuning
│   ├── trl/                        # Cloned transformer reinforcement learning & self-rewarding
│   ├── repeng/                     # Cloned representation engineering & activation steering
│   └── self-instruct/              # Cloned self-instruct bootstrapping engine
├── papers/
│   ├── PAPERS_INDEX.md             # Complete literature taxonomy & mapping
│   ├── SEAL_and_Recent_Literature.md # MIT SEAL, MoE-CL, MAD Nature 2024, & SIA
│   ├── ROME_and_MEMIT.md           # Factual editing mathematical breakdown
│   ├── Task_Arithmetic_and_Weight_Vectors.md # Weight vector algebra & DARE formulation
│   ├── Self_Rewarding_and_SPIN.md  # DPO & self-rewarding mathematical breakdown
│   └── Automated_Self_Evolution_AI_Scientist.md # Topology & code self-mutation
├── tests/
│   └── test_pipeline.py            # Comprehensive 11-test unit suite
├── requirements.txt
└── README.md
```

---

## ⚡ Quickstart

```bash
# Run full unit tests
python -m unittest discover -s tests

# Run autonomous self-evolution pipeline
python -m ouroboros.cli --generations 3 --domain math_logic
```
