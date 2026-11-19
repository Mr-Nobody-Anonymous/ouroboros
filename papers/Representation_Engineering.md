# Representation Engineering: Top-Down Model Transparency & Activation Steering

- **Seminal Papers**:
  - *Representation Engineering: A Top-Down Approach to AI Transparency and Control* (Zou et al., Center for AI Safety & CMU, 2023) — [arXiv:2310.01405](https://arxiv.org/abs/2310.01405)
  - *Steering Language Models with Activation Addition* (Turner et al., 2023) — [arXiv:2308.10248](https://arxiv.org/abs/2308.10248)
- **Primary Breakthrough**:
  - Rather than updating billions of static weights on disk, the model's behavior, truthfulness, reasoning stance, and safety can be controlled dynamically by adding "steering vectors" directly into intermediate residual stream activations during the forward pass:
    $$x_l' = x_l + c \cdot v_{steer}$$
  - Enables non-destructive cognitive reprogramming in real-time.

---

## 🧮 Mathematical Formulation

### 1. Contrastive Difference-in-Means
Given paired contrastive prompts (e.g. Truthful vs. Hallucinatory, Rigorous vs. Sloppy):
$$v_{raw} = \frac{1}{|D_+|} \sum_{i \in D_+} a_l(x_i^+) - \frac{1}{|D_-|} \sum_{j \in D_-} a_l(x_j^-)$$
where $a_l(x)$ denotes the activation at layer $l$ on the target token.

### 2. PCA Reading Vector
Extracting the first principal component of contrastive pairs:
$$v_{PCA} = \text{PCA}_1 \left( \{ a_l(x_i^+) - a_l(x_i^-) \}_{i=1}^N \right)$$

### 3. Layer Injection
At inference or during fine-tuning:
$$a_l^{(modified)} = a_l + \gamma \cdot \frac{v_{PCA}}{\|v_{PCA}\|_2}$$
where $\gamma$ is a scalar steering strength coefficient.

---

## 💻 Integrated in Ouroboros
- Implemented in `ouroboros/core/architecture_mutator.py` (`compute_steering_vector`)
- Linked to external `external/repeng/`
