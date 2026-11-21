"""Run one new explicitly synthetic research-foundation integration fixture, never a benchmark.

No pretrained downloads. Requires the train extra; leaves real model/training/evaluation
artifacts under a new workdir. The 135M/360M discovery studies are never resumed or touched.
"""

import argparse
import json
from pathlib import Path

from ouroboros.core.checkpoints import file_sha256
from ouroboros.research.controller import run_research
from ouroboros.research.project import init_foundation, verify_run
from ouroboros.research.provenance import model_inventory, write_once


def create_random_model(directory):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    vocabulary = {
        word: index
        for index, word in enumerate(
            [
                "[PAD]",
                "[UNK]",
                "[EOS]",
                "Instruction",
                ":",
                "Response",
                "Return",
                "ready",
                "only",
                ".",
                "train",
                "practice",
                "validation",
                "reserved",
                "final",
                "retention",
                "followup",
                "record",
                *map(str, range(12)),
            ]
        )
    }
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]"
    )
    fast.save_pretrained(directory)
    with torch.random.fork_rng():
        torch.manual_seed(7)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=len(vocabulary),
                n_positions=128,
                n_embd=24,
                n_layer=1,
                n_head=2,
                resid_pdrop=0,
                embd_pdrop=0,
                attn_pdrop=0,
                bos_token_id=2,
                eos_token_id=2,
                pad_token_id=0,
            )
        )
    model.save_pretrained(directory, safe_serialization=True)
    return {
        "initialization_seed": 7,
        "parameters": sum(p.numel() for p in model.parameters()),
        "model_files": model_inventory(directory),
        "pretrained": False,
        "task": "constant ready answer, deliberately trivial, no scientific benchmark claim",
        "unseen_semantic_generalization_demonstrated": False,
    }


def fixture_config(workdir):
    path = Path(init_foundation(workdir / "project", workdir / "model")["config"])
    config = json.loads(path.read_text())
    config["budget"].update(
        max_seconds=600,
        max_seq_length=64,
        max_rss_bytes=1500 * 1024**2,
        cpu_threads=1,
        max_steps=50,
        checkpoint_every=50,
    )
    config["followup_recipe"]["target_modules"] = ["c_attn"]
    path.write_text(json.dumps(config, indent=2) + "\n")
    return path


def run(workdir):
    workdir = Path(workdir).absolute()
    workdir.mkdir(parents=True, exist_ok=False)
    model = create_random_model(workdir / "model")
    write_once(workdir / "fixture_model.json", model)
    config = fixture_config(workdir)
    report = run_research(config)
    verified = verify_run(report["report_file"])
    result = {
        "schema_version": 1,
        "kind": "research-foundation-synthetic-integration",
        "script_sha256": file_sha256(__file__),
        "model": model,
        "run_id": report["run_id"],
        "report_file": report["report_file"],
        "report_sha256": file_sha256(report["report_file"]),
        "state": report["state"],
        "full_gate_passed": report["full_gate_passed"],
        "baseline_practice": report.get("baseline_practice"),
        "candidate_practice": [
            {
                "recipe": c["recipe"],
                "scores": [r["practice"] for r in c["replications"]],
                "prediction_audit": c["prediction_audit"],
            }
            for c in report["candidates"]
        ],
        "gates": report["gates"],
        "artifact_verification": verified,
        "independent_workers": len(report["workers"]),
        "source_files_unchanged": report["source_files_unchanged"],
        "autonomous_improvement_demonstrated": False,
        "pretrained_gain_demonstrated": False,
        "improved_future_selection_demonstrated": False,
        "note": "Host-authored full-SFT/LoRA controls on an extremely small random LM. Constant-answer prompts are deliberately easy and not evidence of reasoning or broad generalization.",
    }
    write_once(workdir / "fixture_summary.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args(argv)
    result = run(args.workdir)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["state"] == "completed" and result["source_files_unchanged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
