# Editing Models with Task Arithmetic & Deep Weight Merging

- **Papers**:
  - *Editing Models with Task Arithmetic* (Ilharco et al., 2022) — [arXiv:2212.04089](https://arxiv.org/abs/2212.04089)
  - *Deep Model Merging without Training: TIES-Merging & DARE* (Yadav et al., 2023; Yu et al., 2023) — [arXiv:2311.03099](https://arxiv.org/abs/2311.03099)
- **Primary Breakthrough**:
  - Fine-tuning creates a "task vector" $\tau_t = \theta_t - \theta_{pre}$ in weight space.
  - Weights can be mathematically added, subtracted (unlearning negative behaviors/biases), multiplied, and merged across models without retraining!

---

## 🧮 Mathematical Operations on Weight Space

### 1. Task Vector Addition & Scaling
To transfer a capability learned by fine-tuned model $\theta_A$ to baseline $\theta_0$:
$$\theta_{new} = \theta_0 + \lambda \tau_A = \theta_0 + \lambda (\theta_A - \theta_0)$$

### 2. Task Unlearning (Negative Vectors)
To eradicate unwanted behaviors, toxic patterns, or hallucinations:
$$\theta_{unlearned} = \theta_0 - \beta \tau_{negative}$$

### 3. TIES-Merging (Trimming, Elect Sign, Disjoint Merge)
When merging multiple self-fine-tuned models $\tau_1, \tau_2, \dots, \tau_K$:
1. **Trim**: Retain only top-$p\%$ magnitude weights (sparsification).
2. **Elect Sign**: Resolve directional conflicts via consensus majority: $\text{sign}(\gamma_m) = \text{sign}\left(\sum_{k=1}^K \hat{\tau}_k^{(m)}\right)$.
3. **Disjoint Merge**: Average only matching sign parameters.

### 4. DARE (Drop And REscale)
Randomly zeroes out $\rho \in [0.90, 0.99]$ of weight deltas and rescales by $\frac{1}{1 - \rho}$:
$$\Delta W_{DARE} = \frac{m \odot \Delta W}{1 - \rho}, \quad m \sim \text{Bernoulli}(1 - \rho)$$

---

## 💻 Integrated in Ouroboros
- Implemented in `ouroboros/core/task_arithmetic.py`
- Bridged to MergeKit in `ouroboros/integrations/mergekit_bridge.py`
