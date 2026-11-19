# ⚡ Sparks of Weight-Space Programming: LLMs Can Natively Read & Write Neural Network Weights

- **Authors**: Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab (MIT CSAIL, Nov 2026)
- **Title**: *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*
- **Primary Breakthrough**:
  - Neural network weights are traditionally "write-only" artifacts produced by gradient descent and backpropagation.
  - Demonstrates that frontier LLMs (such as **GPT-6 Astra**) can natively inspect, interpret, and directly edit raw weights of target models like **Qwen3.5-0.8B** without traditional training, backpropagation, or optimizer passes.

---

## 🎯 The Seminal Qwen3.5-0.8B Mathematical Surgery

### 1. The Glitch Diagnosis
A small locally running model (**Qwen3.5-0.8B**) was trapped by repetitive reasoning loops during multi-step mathematical problems, limiting its baseline **MATH-500** benchmark performance to just **7.9%**.

### 2. Surgical Parameter Surgery
* **Zero Gradient Descent**: GPT-6 Astra was given direct programmatic read/write access to Qwen's parameter tensors via Python.
* **Directional Ablation in Weight Space**: Astra diagnosed the looping mechanism in the down-projection MLP layers and surgically altered just **0.016% of total parameters** (~131,000 weights out of 800 million).
* **The Result**:
  * **MATH-500 Score surged from 7.9% to 48.1%** (~6x improvement, reaching the 40%–50% range).
  * **IFEval (Instruction Following)** tripled as a byproduct of fixing the repetitive attractor state.

---

## 🧮 Mathematical Formulation of Weight-Space Surgery

### 1. Attractor Direction Identification
Let $W_{down} \in \mathbb{R}^{d_{model} \times d_{ffn}}$ be the MLP down-projection matrix. A degenerate repetitive attractor is represented by top singular vector $u_{loop}$:
$$u_{loop} = \arg\max_{\|u\|=1} \| W_{down}^T u \|$$

### 2. Directional Ablation
Rather than retraining the layer, the projection onto the degenerate subspace is attenuated:
$$W_{down}' = W_{down} - \eta (u_{loop} u_{loop}^T) W_{down}$$
where $\eta \in (0, 1]$ controls the ablation strength.

### 3. Sparse Weight Masking (0.016% Top Delotas)
To minimize collateral damage to unrelated general capabilities:
$$\Delta W = W_{down}' - W_{down}$$
$$\text{Mask}_{ij} = \mathbb{I}(|\Delta W_{ij}| \ge \text{TopK}_{0.016\%}(|\Delta W|))$$
$$W_{final} = W_{down} + (\text{Mask} \odot \Delta W)$$

---

## 💻 Integrated in Ouroboros
- Implemented natively in `ouroboros/core/weight_space_programmer.py`
- Integrated into `ouroboros/pipeline/self_loop.py`
