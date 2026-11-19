# Self-Instruct & Autonomous Curriculum Generation

- **Seminal Papers**:
  - *Self-Instruct: Aligning Language Models with Self-Generated Instructions* (Wang et al., UW & AI2, 2022) — [arXiv:2212.10560](https://arxiv.org/abs/2212.10560)
  - *UltraFeedback: Boosting Language Models with High-Filtered AI Feedback* (Cui et al., 2023) — [arXiv:2310.01377](https://arxiv.org/abs/2310.01377)
- **Primary Breakthrough**:
  - Showed that pre-trained language models can autonomously bootstrap their own instruction-following ability by generating diverse tasks, input instances, and target outputs from a small seed set of human examples.
  - Paved the way for modern synthetic data generation pipelines (e.g., Alpaca, Orca, UltraFeedback).

---

## 🧮 Autonomous Bootstrapping Pipeline

```
  ┌────────────────────────────────────────────────────────┐
  │ Seed Tasks Pool (e.g., 175 human seed tasks)          │
  └──────────────────────────┬─────────────────────────────┘
                             │ Sample 8 tasks (3 human + 5 model)
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ Task Generation: Prompt model to invent new operations  │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ Input/Output Generation (Input-First or Output-First)  │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ Quality Filtering (ROUGE-L overlap < 0.7, heuristic    │
  │ length checks, self-critique verification)             │
  └──────────────────────────┬─────────────────────────────┘
                             │ Pass
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ Add to Training Pool & Fine-Tune Model Weights         │
  └────────────────────────────────────────────────────────┘
```

---

## 💻 Integrated in Ouroboros
- Implemented in `ouroboros/core/meta_feedback.py`
- Linked to external `external/self-instruct/`
