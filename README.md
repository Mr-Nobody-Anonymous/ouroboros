# 🐍 Ouroboros: Autonomous Recursive Self-Fine-Tuning Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Architecture: PEFT & Model Editing](https://img.shields.io/badge/Architecture-PEFT%20%7C%20ROME%20%7C%20MEMIT-orange)](https://github.com)

**Ouroboros** is an autonomous framework engineered for Large Language Models (LLMs) to inspect, critique, and recursively fine-tune their own mathematical weight matrices. By combining Parameter-Efficient Fine-Tuning (PEFT/LoRA), surgical tensor editing (ROME/MEMIT), and closed-loop anti-catastrophic-forgetting verification, Ouroboros enables models to evolve their own cognitive capabilities across successive generations.

---

## 🏛️ Architecture Overview

```
                      ┌────────────────────────────────────────┐
                      │        Autonomous Self-Diagnosis       │
                      │   (Identify weakest reasoning domain)  │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │         Curriculum Synthesis           │
                      │  (Critique-correction synthetic data)  │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │            Weight Adjustment           │
                      │  • LoRA ΔW = (α/r) · (B · A)           │
                      │  • ROME / MEMIT Surgical Weight Surgery│
                      │  • Norm-bounding & Clamping            │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │     Post-Tuning Verification Suite     │
                      │  (Checks progress & detects forgetting)│
                      └───────────┬────────────────┬───────────┘
                                  │                │
                        [Pass]    ▼                ▼    [Degrade]
                      ┌────────────────┐      ┌────────────────┐
                      │ Save Gen N     │      │ Rollback to    │
                      │ Checkpoint     │      │ Previous Gen   │
                      └────────────────┘      └────────────────┘
```

---

## 📦 Directory Structure

```plaintext
ouroboros-tuner/
├── ouroboros/
│   ├── core/
│   │   ├── weight_editor.py       # Direct tensor surgery, SVD rank updates, LoRA deltas & clamping
│   │   ├── adapter_engine.py      # Low-rank adapter parameter management & PEFT updates
│   │   ├── meta_feedback.py       # Autonomous error diagnosis & synthetic training pair curation
│   │   └── evaluator.py           # Multi-domain benchmark engine & catastrophic forgetting guard
│   ├── pipeline/
│   │   └── self_loop.py           # Recursive generation loop orchestrator
│   ├── integrations/
│   │   └── easyedit_bridge.py     # Seamless bridge to ROME, MEMIT, and MEND model editing
│   ├── config.py                  # Model, tuning, and cycle hyperparameters
│   └── cli.py                     # Command-line interface
├── external/
│   └── EasyEdit/                  # Cloned SOTA knowledge & weight editing toolkit
├── tests/
│   └── test_pipeline.py           # Full unit test suite for tensor math and evaluation
├── requirements.txt               # Dependencies (PyTorch, PEFT, Accelerate, etc.)
└── README.md
```

---

## 🔬 Mathematical Foundations

### 1. Low-Rank Adaptation (LoRA)
Rather than updating full parameter matrices $W_0 \in \mathbb{R}^{d \times k}$, updates are decomposed into:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$
where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d, k)$.

### 2. Surgical Weight Editing (Rank-One Model Editing)
For localized factual and logical corrections without retraining:
$$\Delta W = (v^* - W_0 k^*) \left( \frac{(k^*)^T}{(k^*)^T k^* + \lambda} \right)$$

### 3. Stability Norm Clamping
To prevent catastrophic runaway divergence during recursive iterations:
$$\|\Delta W\|_F \le \tau \cdot \|W_0\|_F$$
Updates exceeding the threshold ratio $\tau$ are scaled down uniformly.

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/your-username/ouroboros-tuner.git
cd ouroboros-tuner
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
python -m unittest discover -s tests
```

### 3. Launch Recursive Self-Tuning Cycle
```bash
python -m ouroboros.cli --generations 3 --domain math_logic
```

---

## 🛡️ Anti-Catastrophic Forgetting
Every evolutionary cycle benchmarks the updated parameters against frozen anchor baselines across math, logic, coding, and instruction following. If any baseline score degrades beyond the safety threshold ($\Delta < -5\%$), the cycle triggers an immediate rollback to the previous generation's checkpoint.
