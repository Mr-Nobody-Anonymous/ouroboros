# Automated AI Discovery & Architecture Evolution (The AI Scientist & Self-Modifying Code)

- **Papers**:
  - *The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery* (Lu et al., Sakana AI & Oxford, 2024) — [arXiv:2408.06292](https://arxiv.org/abs/2408.06292)
  - *Large Language Models as Tool Makers* (Cai et al., 2023) — [arXiv:2305.17126](https://arxiv.org/abs/2305.17126)
  - *Darwinian Evolution of LLM Topologies* (Sakana AI / Evolutionary Model Merging, 2024)
- **Primary Breakthrough**:
  - Demonstrates that language models can autonomously modify Python codebases, alter neural network hyperparameters, prune layers, inject cross-attention pathways, execute experimental benchmarks, write papers, and review their own scientific contributions.

---

## 🧮 Architectural Mutation Mechanisms

### 1. Structural Layer Surgery
- **Frankenmerging / Layer Stacking**: Repeating or reordering transformer blocks to expand reasoning depth without training from scratch.
- **Attention Head Reweighting**: Applying dynamic scalar multipliers $\gamma_h$ to multi-head self-attention outputs:
  $$\text{MHA}(Q, K, V) = \sum_{h=1}^H \gamma_h \cdot \text{Head}_h W_O$$

### 2. Autonomous Code Mutation Loop
1. **Hypothesis Generation**: Propose structural or algorithmic optimization (e.g. KV-cache compression, custom loss function).
2. **Code Patching**: Self-generate diffs to runtime execution scripts.
3. **Sandbox Validation**: Run automated test suite to ensure stability.
4. **Verification**: Measure performance improvements against baseline metrics.

---

## 💻 Integrated in Ouroboros
- Implemented in `ouroboros/core/architecture_mutator.py`
