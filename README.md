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
   * *The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery* ([arXiv:2408.06292](https://arxiv.org/abs/2408.06292))
   * *Darwinian Evolution of LLM Topologies* (Sakana AI, 2024)

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
│   │   └── self_rewarding.py       # Self-rewarding rubric, LLM-as-a-judge, DPO dataset generation
│   ├── pipeline/
│   │   └── self_loop.py            # Recursive evolution loop orchestrator
│   ├── integrations/
│   │   ├── easyedit_bridge.py      # Interface to EasyEdit (ROME, MEMIT, MEND)
│   │   └── mergekit_bridge.py      # Interface to MergeKit (TIES, DARE, SLERP recipes)
│   ├── config.py                   # Configuration dataclasses
│   └── cli.py                      # Command-line entrypoint
├── external/
│   ├── EasyEdit/                   # Cloned knowledge editing repository
│   └── mergekit/                   # Cloned model weight merging & arithmetic repository
├── papers/
│   ├── PAPERS_INDEX.md             # Complete literature taxonomy & mapping
│   ├── ROME_and_MEMIT.md           # Factual editing mathematical breakdown
│   ├── Task_Arithmetic_and_Weight_Vectors.md # Weight vector algebra & DARE formulation
│   ├── Self_Rewarding_and_SPIN.md  # DPO & self-rewarding mathematical breakdown
│   └── Automated_Self_Evolution_AI_Scientist.md # Topology & code self-mutation
├── tests/
│   └── test_pipeline.py            # Comprehensive 9-test unit suite
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
