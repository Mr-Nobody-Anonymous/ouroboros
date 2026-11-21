"""Source-linked DeepSeek review: what was adopted, what was refused, what stays deferred.

Every entry names the official repository (pinned in the review cache), the mechanism, and
the local evidence. `status` is deliberately narrow:

  implemented  - real owned code exists and is exercised by tests
  partial      - a bounded subset exists; the listed differences are material
  deferred     - method understood, no local implementation, no local claim
  not_applicable - requires an architecture/data regime this single-CPU dense-LM loop lacks

Nothing here is a reproduction. Source-reported gains are never local results.
"""

import json
from pathlib import Path

STATUSES = ("implemented", "partial", "deferred", "not_applicable")

ENTRIES = (
    {
        "id": "deepseekmath-corpus-screening",
        "repo": "deepseek-ai/DeepSeek-Math",
        "source": "https://arxiv.org/abs/2402.03300",
        "reviewed_from": "arxiv-html selected sections + official repo documents",
        "mechanism": "Iterative fastText domain mining and benchmark decontamination by n-gram overlap, then 120B-token math corpus construction.",
        "status": "implemented",
        "adopted": "A stricter local admission guard: normalized word n-gram overlap of questions between visible (train/practice/followup) and hidden (validation/reserved/final/robustness/retention) roles refuses a dataset before any model load.",
        "differences": "No fastText classifier, no 120B-token corpus, no pretraining corpus access, no semantic matching. Only question text is screened; answers are not.",
        "local_evidence": [
            "ouroboros/research/data_quality.py",
            "tests/test_research_data_quality.py",
        ],
        "claims_not_made": [
            "pretraining decontamination",
            "semantic leakage detection",
            "paper-scale corpus curation",
        ],
    },
    {
        "id": "deepseekmath-grpo",
        "repo": "deepseek-ai/DeepSeek-Math",
        "source": "https://arxiv.org/abs/2402.03300",
        "reviewed_from": "arxiv-html sections 4",
        "mechanism": "Group Relative Policy Optimization: critic-free group-relative advantages inside a clipped policy objective with reference KL, trained on sampled rollouts.",
        "status": "partial",
        "adopted": "Outcome-level group standardization is computed and recorded for bounded exact-answer rollouts used as *training-data filters*, with an explicit uninformative-group flag.",
        "differences": "No policy-gradient objective, no token log-probabilities, no clipped ratio, no reference KL, no reward model. Standardizing verified rewards is arithmetic, not optimization.",
        "local_evidence": ["ouroboros/research/evaluation.py", "tests/test_research_foundation.py"],
        "claims_not_made": [
            "GRPO training",
            "reinforcement learning",
            "reward-model training",
            "RL sample efficiency",
        ],
    },
    {
        "id": "deepseek-r1-verified-distillation",
        "repo": "deepseek-ai/DeepSeek-R1",
        "source": "https://github.com/deepseek-ai/DeepSeek-R1",
        "reviewed_from": "cached official PDF (methods)",
        "mechanism": "Correctness-filtered rejection sampling of teacher reasoning, then SFT/distillation of accepted completions.",
        "status": "implemented",
        "adopted": "An explicit `distill_sft` family: an independently configured teacher is sampled on *training* prompts only, each completion is accepted only if it matches the host reference exactly under the same normalizer, and refusals are recorded.",
        "differences": "No reasoning-RL stage, no process reward, no proof of rationale faithfulness. Answer correctness does not certify the reasoning text.",
        "local_evidence": ["ouroboros/research/worker.py", "ouroboros/research/evaluation.py"],
        "claims_not_made": [
            "reasoning-RL reproduction",
            "rationale verification",
            "teacher-superiority",
        ],
    },
    {
        "id": "deepseek-r1-reasoning-rl",
        "repo": "deepseek-ai/DeepSeek-R1",
        "source": "https://github.com/deepseek-ai/DeepSeek-R1",
        "reviewed_from": "cached official PDF (methods)",
        "mechanism": "Cold-start SFT, reasoning RL, rejection-SFT, then broader RL, with rule-based and model-based rewards.",
        "status": "deferred",
        "adopted": "Nothing executed. The rejection/distillation SFT subset above is kept separate from any RL claim.",
        "differences": "No verifier-driven RL, no long-form reasoning training, no RL infrastructure or compute budget.",
        "local_evidence": ["papers/DEEPSEEK_APPLICABILITY.md"],
        "claims_not_made": ["RLVR", "emergent long reasoning", "test-time scaling"],
    },
    {
        "id": "prover-verified-expert-iteration",
        "repo": "deepseek-ai/DeepSeek-Prover-V1.5",
        "source": "https://arxiv.org/abs/2408.08152",
        "reviewed_from": "cached official PDF (methods)",
        "mechanism": "Expert iteration over machine-verified proofs with binary Lean correctness, truncate-and-resume search and moderately solvable prompts for group optimization.",
        "status": "partial",
        "adopted": "The verification-first shape: bounded rollout groups, an independent rule check, rejection of unverified completions, and fresh-worker re-evaluation of any resulting checkpoint.",
        "differences": "The verifier is exact string/normalized answer matching, not Lean. There is no proof state, tactic search, truncate/resume or formal guarantee. Syntax validity is never treated as correctness.",
        "local_evidence": ["ouroboros/research/worker.py", "ouroboros/research/training.py"],
        "claims_not_made": [
            "formal verification",
            "theorem-proving performance",
            "proof-search reproduction",
        ],
    },
    {
        "id": "prover2-subgoal-decomposition",
        "repo": "deepseek-ai/DeepSeek-Prover-V2",
        "source": "https://github.com/deepseek-ai/DeepSeek-Prover-V2",
        "reviewed_from": "cached official PDF (methods)",
        "mechanism": "Verified subgoal decomposition to build coarse-grained proof sketches for reinforcement learning.",
        "status": "deferred",
        "adopted": "Nothing. Subgoal decomposition has no analogue without a proof assistant.",
        "differences": "No Lean, no subgoal prover, no sketch-conditioned policy.",
        "local_evidence": ["papers/DEEPSEEK_APPLICABILITY.md"],
        "claims_not_made": ["subgoal decomposition", "proof-sketch training"],
    },
    {
        "id": "mathv2-learned-verification",
        "repo": "deepseek-ai/DeepSeek-Math-V2",
        "source": "https://github.com/deepseek-ai/DeepSeek-Math-V2",
        "reviewed_from": "cached official PDF (methods)",
        "mechanism": "A learned verifier assigns proof scores, plus meta-verification that an alleged flaw actually exists, to avoid rewarding hallucinated criticism.",
        "status": "deferred",
        "adopted": "Refused rather than approximated: the foundation keeps an exact-answer grader and never lets a model score its own work, so no learned-verifier shortcut exists to misconfigure.",
        "differences": "No learned verifier, no meta-verifier, no proof-score labels.",
        "local_evidence": ["ouroboros/research/evaluation.py", "SECURITY.md"],
        "claims_not_made": ["learned verification", "formal certification", "verifier reliability"],
    },
    {
        "id": "v3-training-recipe",
        "repo": "deepseek-ai/DeepSeek-V3",
        "source": "https://github.com/deepseek-ai/DeepSeek-V3",
        "reviewed_from": "arxiv-html sections + official repo documents",
        "mechanism": "Long-context/data curriculum, constant-then-decay schedules, gradient clipping, checkpoint and optimizer bookkeeping, packed-example masks.",
        "status": "implemented",
        "adopted": "Per-step audited traces (loss, pre-clip gradient norm, clip value, learning rate, example index, completion tokens, wall time, peak RSS), constant or cosine schedules, AdamW/SGD, deterministic ordering from a seeded RNG, and optimizer/CPU-RNG export with an explicit no-mid-step-resume contract.",
        "differences": "No sequence packing, no dual-axis pipeline parallelism, no multi-trillion-token curriculum. The paper's scale-specific hyperparameters are not transplanted to a small CPU model.",
        "local_evidence": [
            "ouroboros/research/training.py",
            "tests/test_research_foundation_ml.py",
        ],
        "claims_not_made": [
            "V3 training reproduction",
            "packed-sequence training",
            "distributed scaling",
        ],
    },
    {
        "id": "v3-mtp-and-moe",
        "repo": "deepseek-ai/DeepSeek-V3",
        "source": "https://arxiv.org/abs/2412.19437",
        "reviewed_from": "arxiv-html sections",
        "mechanism": "Multi-token prediction sharing embeddings/head, auxiliary-loss-free expert bias with a small sequence-balancing loss, MLA attention with decoupled RoPE.",
        "status": "not_applicable",
        "adopted": "Introspection reports MoE/MLA-related config fields when present and refuses to substitute a dense approximation.",
        "differences": "The foundation target is a dense small causal LM. Adding MTP or MoE would be a different architecture experiment with its own budgets and gates.",
        "local_evidence": ["ouroboros/research/worker.py"],
        "claims_not_made": ["MTP", "MoE routing", "MLA", "auxiliary-loss-free balancing"],
    },
    {
        "id": "v2-mla-kv-compression",
        "repo": "deepseek-ai/DeepSeek-V2",
        "source": "https://arxiv.org/abs/2405.04434",
        "reviewed_from": "cached official tech report (methods)",
        "mechanism": "Joint low-rank KV compression, decoupled RoPE and weight absorption for cheap cached inference; device-limited expert routing.",
        "status": "not_applicable",
        "adopted": "Nothing at the weight level. Efficiency is *measured* (tokens/second, wall time, peak RSS) rather than assumed from an architecture paper.",
        "differences": "Requires an MLA model trained from scratch; post-hoc KV compression of a dense GPT-2-class model is not the same method.",
        "local_evidence": ["ouroboros/research/worker.py"],
        "claims_not_made": ["MLA", "KV-cache compression", "inference speedup"],
    },
    {
        "id": "v32-sparse-attention",
        "repo": "deepseek-ai/DeepSeek-V3.2-Exp",
        "source": "https://github.com/deepseek-ai/DeepSeek-V3.2-Exp",
        "reviewed_from": "cached official PDF (methods)",
        "mechanism": "DeepSeek Sparse Attention with a trained lightning indexer and top-k selection, dense warm-up followed by joint sparse continued training.",
        "status": "deferred",
        "adopted": "Nothing. The paper's own status is experimental; pruning attention post hoc is explicitly not this method.",
        "differences": "No indexer training, no sparse attention kernels, no long-context continued training.",
        "local_evidence": ["papers/DEEPSEEK_APPLICABILITY.md"],
        "claims_not_made": ["DSA", "long-context efficiency", "kernel performance"],
    },
    {
        "id": "moe-fine-grained-experts",
        "repo": "deepseek-ai/DeepSeek-MoE",
        "source": "https://arxiv.org/abs/2401.06066",
        "reviewed_from": "cached official PDF (methods)",
        "mechanism": "Split FFNs into finer experts, raise the activated count to preserve compute, and keep always-on shared experts to reduce routed redundancy.",
        "status": "not_applicable",
        "adopted": "Nothing. The foundation cannot convert a dense checkpoint into a genuine MoE without new initialization, data and evaluation.",
        "differences": "Architecture change, not an intervention family the loop may propose.",
        "local_evidence": ["ouroboros/research/contracts.py"],
        "claims_not_made": ["expert specialization", "MoE conversion", "load balancing"],
    },
    {
        "id": "esft-expert-specialization",
        "repo": "deepseek-ai/ESFT",
        "source": "https://github.com/deepseek-ai/ESFT",
        "reviewed_from": "arxiv-html + official repo README",
        "mechanism": "Select task-relevant MoE experts from *training* gate statistics with a cumulative threshold, then tune only those experts while leaving routing intact.",
        "status": "deferred",
        "adopted": "Nothing. Dense LoRA is not ESFT, and the interface reports the target's real architecture instead of implying expert selection.",
        "differences": "Requires MoE gating and per-expert parameter isolation; neither exists on a dense target.",
        "local_evidence": ["docs/RESEARCH_FOUNDATION.md"],
        "claims_not_made": ["ESFT", "expert freezing", "expert-level parameter efficiency"],
    },
    {
        "id": "engram-conditional-memory",
        "repo": "deepseek-ai/Engram",
        "source": "https://github.com/deepseek-ai/Engram",
        "reviewed_from": "cached official PDF (methods)",
        "mechanism": "Canonicalized suffix n-gram hashing into learned embedding tables with a context gate and depthwise causal convolution; predictable indices enable host-memory prefetch.",
        "status": "deferred",
        "adopted": "No model-integrated memory. The append-only SQLite ledger is *experiment* memory (episodic/semantic/strategic records) and is explicitly not Engram.",
        "differences": "Engram is a trainable model component with its own tables and gating; the ledger never enters the forward pass.",
        "local_evidence": ["ouroboros/research/memory.py", "SECURITY.md"],
        "claims_not_made": ["Engram", "conditional memory", "n-gram embeddings"],
    },
    {
        "id": "coder-fim-and-data",
        "repo": "deepseek-ai/DeepSeek-Coder",
        "source": "https://arxiv.org/abs/2401.14196",
        "reviewed_from": "arxiv-html section 3",
        "mechanism": "Dependency-aware repository concatenation, near-deduplication, and fill-in-the-middle training at 50% PSM (16B variant); repository-level window mining.",
        "status": "deferred",
        "adopted": "Only the anti-contamination spirit is reused: exact normalized duplicates and question n-gram overlap are refused. FIM is not implemented, and the loop trains completion-style SFT.",
        "differences": "No repo-level mining, no near-dedup index, no PSM/SPM objective, no 87-language data mixture.",
        "local_evidence": ["ouroboros/research/data_quality.py"],
        "claims_not_made": [
            "FIM training",
            "code completion quality",
            "repo-level decontamination",
        ],
    },
    {
        "id": "coderv2-mixture",
        "repo": "deepseek-ai/DeepSeek-Coder-V2",
        "source": "https://arxiv.org/abs/2406.11931",
        "reviewed_from": "cached official PDF (methods)",
        "mechanism": "60% code / 10% math / 30% natural-language mixture, domain mining, and a 6T-token continued pretraining stage; FIM used in the 16B but not the 236B variant.",
        "status": "not_applicable",
        "adopted": "Nothing at pretraining scale. A `data_reweight` family exists for bounded domain repetition, which is not a mixture re-pretraining stage.",
        "differences": "No continued pretraining, no trillion-token mixture, no per-version FIM differences to reproduce.",
        "local_evidence": ["ouroboros/research/contracts.py"],
        "claims_not_made": [
            "continued pretraining",
            "data mixture ablation",
            "code benchmark gains",
        ],
    },
    {
        "id": "llm-scaling-observations",
        "repo": "deepseek-ai/DeepSeek-LLM",
        "source": "https://arxiv.org/abs/2401.02954",
        "reviewed_from": "arxiv-html section 2-4",
        "mechanism": "Multi-step versus cosine learning-rate comparison, batch-size scaling, gradient clipping, and optimizer/model checkpoint cadence at 2T tokens.",
        "status": "partial",
        "adopted": "Cosine and constant schedules are both first-class and are *measured* on the actual target; clipping and optimizer-state records exist per step.",
        "differences": "The paper's scale-specific conclusions are not transplanted; a 1M-parameter fixture cannot validate a 7B/67B scaling law.",
        "local_evidence": ["ouroboros/research/training.py"],
        "claims_not_made": [
            "scaling-law reproduction",
            "optimal hyperparameters",
            "large-batch training",
        ],
    },
    {
        "id": "vl-hybrid-encoders",
        "repo": "deepseek-ai/DeepSeek-VL",
        "source": "https://arxiv.org/abs/2403.05525",
        "reviewed_from": "arxiv-html selected sections",
        "mechanism": "Hybrid vision encoders, modality warm-up, language-retention mixtures and explicit small-model limits for vision-language training.",
        "status": "deferred",
        "adopted": "Nothing executed. The retention lesson is echoed by this loop's own zero-forgetting gate, which is a text-only accuracy rule and not a vision recipe.",
        "differences": "No vision encoder, image-text data, modality warm-up stage or visual evaluator.",
        "local_evidence": ["papers/DEEPSEEK_APPLICABILITY.md"],
        "claims_not_made": [
            "vision-language understanding",
            "modality warm-up",
            "language retention in VLMs",
        ],
    },
    {
        "id": "vl2-dynamic-tiling",
        "repo": "deepseek-ai/DeepSeek-VL2",
        "source": "https://arxiv.org/abs/2412.10302",
        "reviewed_from": "cached official PDF (methods)",
        "mechanism": "Dynamic image tiling with a global thumbnail, 2x2 pixel shuffle to cut visual tokens, and MoE vision-language decoding.",
        "status": "deferred",
        "adopted": "Nothing executed; listed so token-budget language elsewhere in this repository is not mistaken for visual tiling.",
        "differences": "No vision encoder, tiling pipeline, image data or visual evaluator.",
        "local_evidence": ["papers/DEEPSEEK_APPLICABILITY.md"],
        "claims_not_made": [
            "multimodal understanding",
            "tiling efficiency",
            "visual token reduction",
        ],
    },
    {
        "id": "janus-unified-encoders",
        "repo": "deepseek-ai/Janus",
        "source": "https://github.com/deepseek-ai/Janus",
        "reviewed_from": "cached official report + repo documents",
        "mechanism": "Separate understanding and generation encodings under one autoregressive transformer, with staged training and data mixtures.",
        "status": "deferred",
        "adopted": "Nothing. The separation-of-roles idea appears here as proposer/executor/evaluator process separation, not as a multimodal architecture.",
        "differences": "No image tokenizer, generation head, image data or visual metrics.",
        "local_evidence": ["docs/RESEARCH_FOUNDATION.md"],
        "claims_not_made": [
            "image generation",
            "unified multimodal model",
            "visual instruction tuning",
        ],
    },
    {
        "id": "ocr-optical-compression",
        "repo": "deepseek-ai/DeepSeek-OCR",
        "source": "https://github.com/deepseek-ai/DeepSeek-OCR",
        "reviewed_from": "cached official PDF (methods)",
        "mechanism": "SAM-based convolutional token compression, a CLIP global encoder and an MoE decoder across multiple resolution modes.",
        "status": "deferred",
        "adopted": "Nothing. No document images, OCR corpus or text-recognition metric exists here.",
        "differences": "Requires vision towers and multi-billion-parameter decoding; local 'compression' means checkpoint and output-byte bounding, which is unrelated.",
        "local_evidence": ["papers/DEEPSEEK_APPLICABILITY.md"],
        "claims_not_made": ["OCR quality", "optical compression", "document understanding"],
    },
    {
        "id": "ocr2-visual-causal-flow",
        "repo": "deepseek-ai/DeepSeek-OCR-2",
        "source": "https://github.com/deepseek-ai/DeepSeek-OCR-2",
        "reviewed_from": "cached official PDF (methods)",
        "mechanism": "Replaces the CLIP global encoder with a bidirectional visual prefix from a small causal LM and routes only causal flow queries into the decoder.",
        "status": "deferred",
        "adopted": "Nothing. Recorded so a text-only repository cannot imply a second-generation OCR implementation.",
        "differences": "No visual encoder, query routing, image data or decoding experiments.",
        "local_evidence": ["papers/DEEPSEEK_APPLICABILITY.md"],
        "claims_not_made": ["visual causal flow", "OCR-2 reproduction", "token routing gains"],
    },
)


def report(paths=None):
    """Return the matrix plus integrity facts; optionally check local evidence exists."""
    entries = [dict(entry) for entry in ENTRIES]
    identifiers = [entry["id"] for entry in entries]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate applicability entry")
    for entry in entries:
        if entry["status"] not in STATUSES:
            raise ValueError(f"Unknown status for {entry['id']}")
        if not entry["claims_not_made"] or not entry["local_evidence"]:
            raise ValueError(f"Entry {entry['id']} must state non-claims and local evidence")
        if entry["status"] == "implemented" and not any(
            path.startswith("ouroboros/") for path in entry["local_evidence"]
        ):
            raise ValueError(f"Implemented entry {entry['id']} must cite owned code")
    missing = []
    if paths is not None:
        root = Path(paths)
        missing = sorted(
            {
                path
                for entry in entries
                for path in entry["local_evidence"]
                if not (root / path).exists()
            }
        )
    return {
        "schema_version": 1,
        "kind": "deepseek-source-linked-applicability",
        "repositories_reviewed": len({entry["repo"] for entry in entries}),
        "documented_repositories": 19,
        "counts": {status: sum(e["status"] == status for e in entries) for status in STATUSES},
        "entries": entries,
        "missing_local_evidence": missing,
        "source_reported_gains_reproduced_locally": False,
        "review_is_exhaustive": False,
        "note": "Official pinned repositories and selected method sections were read. Equations/figures lost in text extraction, truncated stdout and unread appendices remain; nothing here is a reproduction.",
    }


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="DeepSeek applicability matrix (no model load)")
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    value = report(args.root)
    print(json.dumps(value, indent=2))
    return 1 if value["missing_local_evidence"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
