# Automated discovery, code proposals and structural candidates

Primary references:

- Lu et al., [The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery](https://arxiv.org/abs/2408.06292), 2024.
- Akiba et al., [Evolutionary Optimization of Model Merging Recipes](https://arxiv.org/abs/2403.13187), 2024.
- Zhang et al., [Darwin Godel Machine: Open-Ended Evolution of Self-Improving Agents](https://arxiv.org/abs/2505.22954), 2025; [official repository](https://github.com/jennyzzt/dgm).
- Xu et al., [Evo2: Evolving the Evolver through Weight-Harness Co-Evolution](https://arxiv.org/abs/2610.13025), October 2026.

These target different improvement surfaces: scientific workflows, parameter/data-flow merging recipes, agent source-code evolution, and harness/policy co-training. They are not interchangeable proof that a deployed LLM autonomously rewrites all its internals successfully.

## Local status

`CodeMutator` changes exactly one AST-identified numeric literal and exports a reviewable patch with source hashes. Strings/comments are not rewritten accidentally, syntactic validity is checked, and no generated Python is executed or merged. `NumericObjective` is a small bounded scalar arithmetic interpreter, not a general autodiff objective compiler or security sandbox.

`ArchitectureMutator` validates layer blueprints, head dimensions/scales, activation vectors and exact-magnitude weight pruning. Its explicit copied Llama/Qwen-style layer-stacking API updates layer counts/cache indices and leaves the original model untouched; an offline tiny-model forward pass is tested. This is a primitive, not complete evolutionary architecture search. Unstructured zeros do not imply neuron removal, smaller dense checkpoints or faster inference.

`external/dgm` is pinned for research and source-inspection purposes; its Docker/API/benchmark-dependent autonomous loop was not executed. Core runs stay on the user's working branch and never create hidden experiment branches. See [architecture](../docs/SELF_EVOLVING_ARCHITECTURE.md), [sandbox policy](../sandbox/README.md), and [security](../SECURITY.md).
