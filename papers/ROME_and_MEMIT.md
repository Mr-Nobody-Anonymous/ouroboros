# Locating and Editing Factual Associations & Memory in Transformers (ROME & MEMIT)

- **Papers**: 
  - *Locating and Editing Factual Associations in GPT* (Meng et al., 2022) — [arXiv:2202.05262](https://arxiv.org/abs/2202.05262)
  - *Mass-Editing Memory in a Transformer* (Meng et al., 2022) — [arXiv:2210.07229](https://arxiv.org/abs/2210.07229)
- **Primary Breakthrough**:
  - Identified that Transformer Feed-Forward Network (FFN/MLP) layers act as linear associative memories (keys = subject activations, values = object predictions).
  - Demonstrates that weights can be directly mathematically edited using closed-form linear algebra without full backpropagation.

---

## 🧮 Mathematical Formulation

### 1. Causal Tracing
Through corrupted input generation and hidden state restoration, ROME isolates the exact layer $l^*$ and token position where key associative memories are recalled.

### 2. Rank-One Update (ROME)
For a key vector $k$ (subject representation) and target value vector $v$ (desired prediction):
$$\Delta W = (v - W_0 k) \cdot \frac{k^T C^{-1}}{k^T C^{-1} k}$$
where $C = \mathbb{E}[k k^T]$ is the uncentered covariance matrix of activations over a reference corpus to preserve unedited facts.

### 3. Multi-Layer Mass Editing (MEMIT)
MEMIT scales ROME from single edits to thousands of simultaneous edits by distributing the residual vector across multiple sequential transformer MLP layers:
$$\Delta = v - W_0 k$$
$$\Delta W_l = R_l K_l^T (C_l + K_l K_l^T)^{-1}$$
where $R_l$ represents the residual error allocated to layer $l$.

---

## 💻 Integrated in Ouroboros
- Implemented in `ouroboros/core/weight_editor.py` (`surgical_rank_one_edit`)
- Bridged to EasyEdit in `ouroboros/integrations/easyedit_bridge.py`
