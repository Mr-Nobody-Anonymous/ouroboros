# LoRA and QLoRA

- Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685), 2021.
- Dettmers et al., [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314), 2023.

For a frozen matrix W, LoRA learns `delta_W = (alpha/r) B A`. The added parameter count is `r*(d_in+d_out)` rather than `d_in*d_out`. Initializing B to zero preserves the starting function. This is an empirically useful low-rank adaptation assumption, not a proof that every possible update is low rank.

QLoRA combines low-rank adapters with quantized frozen weights and memory-saving techniques. Published hardware/quality results depend on model, sequence length, optimizer, precision and implementation; they are not universal guarantees that a 70B model fits a particular GPU without degradation.

## Local status

- The NumPy adapter/reference backend computes real analytical gradients for a tiny **classifier**, exports tensors and tests reload/rollback.
- The Transformers backend uses installed PEFT for real completion-masked causal-LM SFT. Offline CPU tests construct a tiny GPT-2, verify only adapters change, and check safetensors export/reload and predictions.
- Optional 4-bit QLoRA requires a local model, CUDA and bitsandbytes. **The GPU/quantization path was not tested in this environment.**
- The default controller does not claim a LoRA gradient step plus a random score bump is LLM self-improvement.

See [local training](../docs/TRAINING.md) and [test evidence](../docs/TESTING.md). Base model weights must remain available; an adapter checkpoint is not a standalone foundation model.
