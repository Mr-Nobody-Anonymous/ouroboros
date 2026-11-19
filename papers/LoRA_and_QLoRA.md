# Low-Rank Adaptation & Quantized Low-Rank Adaptation (LoRA & QLoRA)

- **Seminal Papers**:
  - *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., Microsoft, 2021) — [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
  - *QLoRA: Efficient Finetuning of Quantized LLMs* (Dettmers et al., UW, 2023) — [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- **Primary Breakthrough**:
  - Proved that weight updates during adaptation have a low intrinsic dimension.
  - Frozen base model weights $W_0 \in \mathbb{R}^{d \times k}$ are injected with trainable rank decomposition matrices:
    $$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x$$
  - QLoRA introduces NormalFloat4 (NF4) quantization, Double Quantization, and Paged Optimizers, allowing 65B/70B models to fine-tune on a single 48GB GPU without performance degradation.

---

## 🧮 Mathematical Details

### 1. Intrinsic Rank Factorization
For rank $r \ll \min(d, k)$:
- $A \sim \mathcal{N}(0, \sigma^2)$ is initialized with random Gaussian noise.
- $B = 0$ is initialized to zeros, ensuring $\Delta W = 0$ at initialization ($t = 0$), so model behavior starts identical to the base pre-trained model.

### 2. Zero Inference Latency Folding
Once fine-tuning converges, the low-rank delta is merged directly into the base weights:
$$W_{merged} = W_0 + \frac{\alpha}{r} (B \cdot A)$$
This requires zero architectural changes or extra compute during subsequent inference.

---

## 💻 Integrated in Ouroboros
- Implemented natively in `ouroboros/core/adapter_engine.py` and `ouroboros/core/weight_editor.py`
- Linked to external `external/peft/`
