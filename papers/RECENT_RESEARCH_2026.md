# Recent research: August 19–November 19, 2026

**Checked 2026-11-19.** Eleven selected sources were first released inside this three-month window. This is not every relevant paper on the internet. All experimental figures below are **reported by the cited authors**, not reproduced by Ouroboros. Most entries were reviewed at the primary abstract level; the MIT post's math experiment and selected self-rewriting methods/figures were also inspected. No assertion of a full systematic literature review or independent benchmark replication is made.

The [machine-readable catalogue](research.json) records dates, source types, review scope, local mapping and reproduction=false. [BibTeX](references.bib) uses abbreviated author lists for long collaborations. Links below point to primary sources, not news summaries. Original notes are stored here instead of unlicensed PDF copies.

## 1. A roadmap, not a finished recursive optimizer

**The Last AI Built by Humans: Toward Genuine Recursive Self-Improvement** — Yi Duan et al.; first submitted September 10, updated September 22. [arXiv:2609.11873](https://arxiv.org/abs/2609.11873).

The paper distinguishes autonomy in executing improvements, choosing strategies, acquiring experience, adapting to environments, and improving the improvement process. This taxonomy helps separate an agent changing prompts from one persistently changing weights or its own optimizer. It is a roadmap/survey with preliminary evidence, not a recipe that guarantees unbounded improvement. **Local consequence:** the architecture document identifies which loops are implemented and which remain research references; Ouroboros does not claim the final meta-improvement level.

## 2. Updating both harness and policy

**Evo2: Evolving the Evolver through Weight-Harness Co-Evolution** — Chejian Xu et al.; October 9. [arXiv:2610.13025](https://arxiv.org/abs/2610.13025).

A shared policy acts as proposer and reflector; harness edits and their outcomes feed SFT plus interleaved GRPO training through EvoTIDE. This connects code/scaffold changes to persistent policy learning rather than treating them as independent knobs. The paper reports coding-benchmark improvements in its particular setup. **Limit:** exporting a syntactically valid patch is not reproducing EvoTIDE or demonstrating those gains. **Local consequence:** keep source proposals and measured adapter experiments separate, record evidence, and require review before trusted code changes. No official implementation was added without verifying its availability.

## 3. Continual adaptation still forgets

**Are Self-Evolving Agents Continual Learners?** — Xiaoyin Chen et al.; October 19. [arXiv:2610.25858](https://arxiv.org/abs/2610.25858).

Tests nonparametric self-evolving agents on drifting task streams. Global prompt/workspace changes can interfere with older or unseen tasks; retrieval localizes some effects but can collide on similar queries with conflicting workflows. Better aggregate scores do not establish continual-learning stability. **Local consequence:** compare each candidate against both the original baseline and latest accepted model, preserve verified replay, and insist on separate final tests for generalization. Our engineering gate is not a reproduction of the paper's full evaluation.

## 4. The Astra/Qwen result the user described

**Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights** — Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab, **MIT CSAIL**, November 2. [Primary research post](https://rikiyat.github.io/blog/weight-space-programming/); [released logs](https://github.com/RikiyaT/lm_understand-logs).

This is a research **blog post**, not a verified Harvard/arXiv paper. With a Python environment and 50 allowed forward passes, the authors report GPT-6 Astra editing 131,072 entries (about 0.016%) of Qwen3.5-0.8B, taking MATH-500 from **7.9% to 48.1%** and IFEval from 21.0% to 62.3%. That is a **40.2 percentage-point** math gain, about 6.09 times baseline accuracy, not a 40% relative increase.

The post describes contrastive productive/repetitive activations, layer/strength experiments, a norm-matched random-direction control, and changes to selected MLP output columns in layers 15 and 19. **It does not say that taking the largest singular vector of any weight matrix diagnoses a looping circuit.** The Gemma 3 270M follow-up did not establish a comparable improvement. Fixing a failure that suppresses an existing skill is not proof of learning arbitrary new skills without training.

**Local consequence:** cloned/read pinned logs/code; [Research Lab](../docs/RESEARCH_LAB.md) now adds measured hidden/MLP/logprob/attention reads, contrast directions, column ranking, multi-layer numerical programs, norm-matched/no-op/layer controls and strength sweeps. Legacy [one-matrix trials](../docs/WEIGHT_EDITING.md) remain separate. Numeric inspection is not semantic understanding. Real 135M model-authored attempts failed; separately scripted integration plateaued/rejected. No claim to reproduce Astra's reasoning, exact patch, games/diffusion/Go experiments or benchmark is made; [feature audit](../docs/PAPER_PARITY.md). See [the detailed correction](Sparks_of_Weight_Space_Programming.md).

## 5. Counterevidence on reading semantics from raw weights

**Can an LLM Actually Reason and Program about Neural Network's Weights or is it just memorization on steroids? Exploring “Weight-Space Programming”** — Nazmus Sakib Ahmed; November 5. [arXiv:2611.08070](https://arxiv.org/abs/2611.08070).

An exploratory audit gives raw small-CNN weights to a model without tools or dataset names. Local statistics can be calculated, but dataset identification and class interpretations rely on familiar priors; label permutation breaks semantic judgments. The author explicitly limits conclusions to mostly single-run exploratory evidence and does not refute all possible editing results. **Local consequence:** do not label SVD output as a semantic diagnosis. Require measured activation evidence, controls and behavioral verification, not plausible narrative alone.

## 6. Sparse edits can help, and abstention matters

**Exploring Weight-Space Programming: Capabilities, Generalization, and Limitations** — Junru Hu et al.; November 6. [arXiv:2611.09382](https://arxiv.org/abs/2611.09382); [official repository](https://github.com/WSP-Scope/WSP-Scope-Project).

Studies 16 trajectories across seven benchmarks and two Qwen model sizes. The abstract reports a 29.45-point MATH-500 gain from 1,023 parameter changes, uneven transfer, and **six trajectories retaining original weights** when improvements cannot be reliably validated. This supports abstention as a valid outcome, not promotion of every generation. **Local consequence:** exact sparse budgets, original/latest checks and actual rollback. **Repository caveat:** at the locked revision the official checkout contains only a paper PDF—no runnable editing implementation or license grant. It is indexed as paper-only, not counted as a working integration.

## 7. Transparent computations, not general LLM self-repair

**Interpretable weight-space programming builds transparent transformers and brain encoding models** — Yash Mehta et al.; November 13. [arXiv:2611.18114](https://arxiv.org/abs/2611.18114).

An agent explicitly codes weights/features for interpretable compact networks and scientific encoding models without backpropagation. The setting rewards human-interpretable computation design and selected task prediction. **Limit:** this does not establish automatic architecture improvement of arbitrary pretrained LLMs or universal interpretability. **Local consequence:** keep structural candidates explicit, copied, shape-checked and separately evaluated. No iWSP algorithm or neuroscience experiment was reproduced.

## 8. Independent evaluation instead of headline chasing

**WeightBench: Evaluating Language Models as Weight-Space Programmers** — Aayush Marasini, Barsat Khadka, Nick Rahimi; November 13. [arXiv:2611.18553](https://arxiv.org/abs/2611.18553); [official code](https://github.com/Aayush-Marasini/WeightBench).

Defines 42 tasks across reading, fixing, writing, recognizing impossible edits and data-free compression. The abstract reports that current agents can handle some small control tasks yet struggle with language-model patching and distinguishing impossible from merely hard changes. Its harness is a separate OS-level execution boundary with metered model calls. **Local consequence:** pin the benchmark source, do not equate unit-test success with its scores, and never claim syntactic checks are a sandbox. No paid agent/benchmark run was executed here.

## 9. Replication narrows the interpretation

**Replicating and Extending Weight-Space Programming: Frontier LLMs Compile Programs into Weights and Read Input Layers** — Roksana Goworek, Alexander Hickey, Federico Nanni, Martin O'Reilly; November 16. [arXiv:2611.21449](https://arxiv.org/abs/2611.21449).

Replicates small-network weight writing and studies modified tasks, decompiled policies, unfamiliar games and model auditing. The authors interpret much of the behavior as program-to-weight compilation and shallow input-layout reading, not unrestricted semantic understanding. **Local consequence:** include independent counterevidence and equal-compute/programmatic controls; a successful numerical patch does not justify a broad claim that an AI understands every weight. This reference is not a claim that Ouroboros replicated the replication.

## 10. A model edits itself, but scale and held-outs matter

**Language Models Can Rewrite Their Own Weights** — Jiajun Fan et al.; November 16, **v2 November 18**. [arXiv:2611.21885v2](https://arxiv.org/abs/2611.21885v2).

The paper reports an open 27B model diagnosing and editing itself without gradients, gaining 6.6 accuracy points on held-out math, and repairing a damaged copy. Smaller sizes abstain or break themselves. Reported sessions restart from base weights; they do not establish cumulative improvement across sessions. The selected methods/figures describe practice checks, hidden-state comparisons and separate evaluation; repeated practice selection is not an untouched final test. **Local consequence:** the new [frozen-editor Research Lab](../docs/RESEARCH_LAB.md) implements many selected methods/appendix contracts: original/edited target tools, model-authored numeric actions, multi-tensor/1D/higher-rank flat edits, paired seeded practice, problem-clustered SE/sign flips, saved alternatives, independent base-reset sessions, once-only sealed reserved validation, confirm/withdraw, private damage and final-test isolation. It is a closed data-only language, **not** the paper's general Python sandbox, vLLM/hybrid/vision stack or full evaluation protocol. The actual tiny model produced zero valid actions; no 27B experiment/gain was reproduced, and no official author code checkout was verified. [Every gap/disposition](../docs/PAPER_PARITY.md).

## 11. Architecture depends on how parameters are written

**Weight-Space Programming Doesn't Need Transformers** — Hyunjong Ok, Jaeho Lee; November 17, work in progress. [arXiv:2611.22055](https://arxiv.org/abs/2611.22055); [linked code](https://github.com/effl-lab/Weight-Space-Programming).

When allowed to design structures for directly written parameters, agents construct sparse associative tables. The paper explicitly notes that pretrained LLMs remain more accurate; it compares structures for direct parameter writing, not all of modern pretraining. **Local consequence:** distinguish interpretable table/circuit synthesis from pretrained Transformer adaptation. The repository is linked for follow-up but was not cloned indiscriminately or claimed as integrated.

## Synthesis: what is still missing from a full autonomous research system?

1. A trustworthy task-specific generator/verifier setup; no universal self-judge can certify every answer.
2. Scientific evaluation with unseen final tests, uncertainty, multiple seeds, compute accounting, locality/safety checks and negative results.
3. A genuinely learned proposal policy and outer adaptation loop; having callbacks is not SEAL/EvoTIDE training.
4. OS-level isolation for untrusted code and strict separation of proposal, grader, credentials and checkpoint authority.
5. Model-specific architecture/quantization compatibility and enough RAM/VRAM/storage for real experiments.
6. Verified artifact availability and redistribution rights. A paper link, gitlink or LFS pattern does not supply absent weights/data.

These are explicit limits and follow-up work, not capabilities hidden behind optimistic names. See [the implemented architecture](../docs/SELF_EVOLVING_ARCHITECTURE.md), [test evidence](../docs/TESTING.md), and [audit corrections](../docs/AUDIT_AND_MIGRATION.md).
