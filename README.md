<p align="center">
  <img src="docs/assets/ouroboros.png" alt="Ouroboros: a circuit-patterned serpent biting its tail" width="230">
</p>

<h1 align="center">Ouroboros</h1>
<p align="center"><strong>A small model. Its own weights. Every edit measured.</strong></p>
<p align="center">Choose → Edit → Evaluate → Keep or Undo</p>

Ouroboros is a local research toolkit for bounded model adaptation. A small LLM can use **its own logits to choose an edit to its own weights**; a separate, deterministic controller measures the candidate, rejects plateaus/regressions, and records what happened. No API key or GPU is required for the CPU quickstart.

**Not magic self-improvement:** the edit menu is host-defined, generated Python is never executed, and a completed experiment can correctly accept **nothing**. Existing LoRA training, replay, reversible patches and source/research tools remain available.

[Self-edit guide](docs/SELF_EDITING.md) · [MCP](docs/MCP.md) · [Architecture](docs/SELF_EVOLVING_ARCHITECTURE.md) · [Test evidence](docs/TESTING.md) · [Research](papers/PAPERS_INDEX.md)

## Run a small LLM

Python **3.10–3.12**, Git, a few GB of free RAM, and space for ML dependencies are recommended for the tested CPU setup. The installer creates a local virtual environment; it does not change system Python or download a model.

```bash
git clone --single-branch --branch arena/4dd3ddb6-ouroboros https://github.com/Mr-Nobody-Anonymous/ouroboros.git
cd ouroboros
python scripts/install.py --profile llm
source .venv/bin/activate  # Windows: .venv\Scripts\activate

ouroboros init ouroboros-lab
# Explicit download: pinned, hash-checked 98 MB pretrained GGUF from GitHub.
# Converted locally to ~542 MB float32 safetensors; no upstream plugin code is run.
ouroboros models fetch-smoke --directory ouroboros-lab/models/smollm2
ouroboros self-edit --config ouroboros-lab/self_edit.json --check-only
ouroboros self-edit --config ouroboros-lab/self_edit.json
```

This uses **SmolLM2-135M-Instruct**, dequantized from a Q4_1 distribution—not the original full-precision checkpoint. The source commit, GGUF checksum, conversion and output hashes are recorded. Model weights/caches stay out of Git. Review the Apache-2.0 model license before reuse. See [installation, original Hub weights, and troubleshooting](docs/SELF_EDITING.md).

The current **0.4 development code lives on the linked branch / [PR #1](https://github.com/Mr-Nobody-Anonymous/ouroboros/pull/1)**, not the historical default branch or the existing v0.3 release. Install this source rather than an unrelated similarly named PyPI package.

### What happens during a run?

1. Validate disjoint practice, validation and final-test files; load only local safetensors.
2. Measure a real validation baseline.
3. Let **the target model itself** select one of a few bounded MLP ablations.
4. Check actual changed-entry/norm/patch-byte budgets and original/latest retention.
5. Keep only strictly improved candidates in memory; reject and undo the rest.
6. Evaluate the selected state on final-test data once, then **undo the accepted chain too**.

The source model files are never overwritten. Accepted patches, if any, are saved separately and must be explicitly replayed in order. `self_edit.json` contains menu probabilities, actual edit counts, predictions, decisions and full-state restoration fingerprints. Exit 0 means the experiment completed—not that the model improved.

### Actually run here

A real **134,515,008-parameter** pretrained model selected two edits, each changing **884 weights** in one MLP matrix. Both candidates were measured, rejected on a plateau, and restored. **No capability gain or accepted patch is claimed.** The four-example strict exact-match smoke test scored zero even for verbose factually correct answers, so it is not a useful general-capability benchmark. [Recorded results and caveats →](docs/SELF_EDITING.md#observed-pretrained-run)

## Use it from an agent: MCP

The optional official **Model Context Protocol** SDK provides a local **stdio** server. The installer above includes it. By default it exposes only dependency/config checks, bounded weight inspection, patch verification and report reading.

```json
{
  "mcpServers": {
    "ouroboros": {
      "command": "/absolute/path/to/ouroboros/.venv/bin/ouroboros-mcp",
      "args": ["--workspace", "/absolute/path/to/ouroboros-lab"]
    }
  }
}
```

Running experiments requires **both** `--allow-experiments` and an explicit `--config self_edit.json` allowlist at server startup. There is no arbitrary shell, model download, patch-apply or delete tool. A subprocess timeout is not an OS security sandbox. [Tools, permissions and client setup →](docs/MCP.md)

## Start lighter, or go deeper

```bash
# Lightweight NumPy classifier demo + MCP, without Torch or any model download:
python scripts/install.py --profile core
# After activating the environment:
ouroboros doctor
ouroboros demo --generations 3  # a color-label classifier, NOT an LLM

# Read-only inspection needs no Torch:
ouroboros weights inspect --file /path/to/model.safetensors --limit 20 --values 8

# Optional real local LoRA SFT:
python scripts/install.py --profile train
# Edit this template to point at your own licensed local model/data:
ouroboros run --config examples/local_sft.json --check-only
ouroboros run --config examples/local_sft.json
```

## Capability status

| Surface | What is implemented |
| --- | --- |
| Same-model editing | Constrained model-selected trials, strict measured gates, reversible accepted chains; not free-form weight-program synthesis |
| Local SFT | Completion-masked Transformers/PEFT LoRA, replay anchors, original/latest retention and adapter rollback; GPU QLoRA untested |
| Weight tools | Bounded safetensors inspection, one-matrix trials and exact target-bound sparse apply/reverse |
| Agent access | JSON CLI, packaged `init`, dependency doctor, read-only-default stdio MCP, explicitly allowlisted execution |
| Other research primitives | Task arithmetic/TIES/DARE/SLERP, copied decoder candidates, explicit EasyEdit/mergekit bridges and review-only source proposals |
| Evidence | Measured predictions, fingerprints, failure reports, separate final test and CI; no invented benchmark gains |

[Training/data formats](docs/TRAINING.md) · [Weight-edit contract](docs/WEIGHT_EDITING.md) · [Integration boundaries](docs/INTEGRATIONS.md) · [Security](SECURITY.md)

## Research and real source checkouts

The [research index](papers/PAPERS_INDEX.md) contains **33 references**, including 11 recent primary-source reviews. Source-reported gains remain separate from local observations. The Astra/Qwen work is an **MIT CSAIL research blog**, not a Harvard paper or an Ouroboros reproduction.

Seven pinned submodules and five preserved snapshots are [inventoried](external/README.md), analyzed in the [repository deep dive](analysis/REPOSITORY_DEEP_DIVE.md), and labeled by actual integration level. Cloning a reference does not automatically install or execute it.

```bash
ouroboros repos sync    # explicit source fetch, not upstream execution
ouroboros repos verify
```

Core operation does not require those checkouts. The additional pretrained smoke source is pinned independently in [model provisioning code](ouroboros/model_files.py); only its model data/license are used.

## Develop and verify

```bash
python scripts/install.py --profile train --dev
python -m pytest -q
ruff check ouroboros tests scripts
ruff format --check ouroboros tests scripts
python scripts/check_docs.py
# Explicit pretrained smoke; downloads only with --fetch:
python scripts/run_pretrained_smoke.py --workdir outputs/pretrained-smoke --fetch
```

[CI](.github/workflows/ci.yml) separates lightweight Python 3.10/3.11/3.12 checks, offline random-LM integration, MCP protocol tests and an explicit pinned pretrained-model smoke run. See [testing](docs/TESTING.md) for observed local/GitHub results and limitations.

## Storage, releases, and license

Models, checkpoints, environments and run outputs are ignored by default. [Storage/LFS guidance](docs/STORAGE_AND_RELEASES.md) covers approved artifacts and safe run-scoped pruning. **No missing historical 600 MB payload has been recovered**, and the artifact registry remains empty; LFS patterns do not create uploads.

The existing [v0.3.0 pre-release](https://github.com/Mr-Nobody-Anonymous/ouroboros/releases/tag/v0.3.0) contains core source/checksums, not these new features, models or upstream trees. Historical releases remain unchanged. [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md) · [Agent guide](AGENTS.md)

Original code: [MIT](LICENSE). Third-party models/sources retain their licenses. The README emblem is AI-generated artwork. Ouroboros is experimental software, not a guarantee of safety, generalization, or indefinite self-improvement.
