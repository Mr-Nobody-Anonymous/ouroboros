<p align="center">
  <img src="docs/assets/ouroboros.png" alt="Ouroboros: a circuit-patterned serpent biting its tail" width="230">
</p>

<h1 align="center">Ouroboros</h1>
<p align="center"><strong>A small model. Its own weights. Every edit measured.</strong></p>
<p align="center">Diagnose → Hypothesize → Edit → Measure → Keep or Undo</p>

Ouroboros is a local research toolkit for bounded model adaptation. Run a small LLM that **chooses edits to its own weights**, or use the new **frozen-editor Research Lab** for model-authored JSON actions, multi-tensor numerical programs, instrumented reads and controlled experiments. Schema-constrained commands and practice-readiness diagnostics make failures inspectable rather than silently manufacturing success. A separate host measures candidates, rejects plateaus/regressions and records the evidence. No API key or GPU is required for the small CPU quickstart.

**Not magic self-improvement:** generated Python is never executed, the tiny pretrained model has **not** discovered a beneficial research edit, and a completed run can correctly accept **nothing**. LoRA training, replay and reversible patches remain separate, usable paths.

[Closed-loop discovery](docs/DISCOVERY_EXPERIMENTS.md) · [Research Lab](docs/RESEARCH_LAB.md) · [Paper gaps](docs/PAPER_PARITY.md) · [Small-LLM guide](docs/SELF_EDITING.md) · [MCP](docs/MCP.md) · [Test evidence](docs/TESTING.md) · [Research](papers/PAPERS_INDEX.md)

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

The current **0.5 development code lives on the linked branch / [PR #1](https://github.com/Mr-Nobody-Anonymous/ouroboros/pull/1)**, not the historical default branch or the existing v0.3 release. Install this source rather than an unrelated similarly named PyPI package.

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

## The paper's missing tools, together in one lab

The old finite-menu loop was **not** the self-rewriting paper's workflow. The [source-linked audit](docs/PAPER_PARITY.md) lists **24 self-rewriting comparisons plus the eight MIT/blog experiments**, marking implemented, partial, missing and unrun separately.

The new `lab` path adds:

- **Frozen editor / edited target separation**, independent base-reset sessions and an operator-private damage fixture.
- **19 research actions / 7 numerical operators:** hidden/MLP/attention/logprob reads, measured contrasts, writer ranking, atomic multi-tensor projections/clipping/entry/bias/low-rank edits, and saved alternatives. Dense flat edits support ranks 1–32; architecture/element limits still apply.
- **Paired seeded sampling**, boxed-math grading, cut/repetition diagnostics, problem-clustered uncertainty, random/no-op/layer controls and strength sweeps.
- **Sealed once-only reserved validation**, confirm/withdraw, a fixed evaluation plan and separate final testing.
- **Full-model-bound exact reversible bundles**, explicit local dataset import, a plan-first external benchmark driver, JSON CLI and allowlisted MCP execution.

```bash
python scripts/install.py --profile research
# Reuse the project/model provisioned above; init also created this config:
ouroboros lab tools
ouroboros lab run --config ouroboros-lab/research_lab.json --check-only
ouroboros lab diagnose --config ouroboros-lab/research_lab.json  # practice only
ouroboros lab run --config ouroboros-lab/research_lab.json
```

**Initial measurement:** the 135M editor made two real generations but produced **0 valid actions**. A separately labelled scripted run completed 16 actions/10 forward requests and changed **1,152 entries in two tensors**; reserved accuracy **0.0→0.0**, rejected/restored. A real official-SDK MCP child also ran the research workflow. [Historical evidence →](docs/RESEARCH_LAB.md)

**Editor/experiment fixes:** optional JSON-schema token filtering (not scripted edit choices), compact reproducible prompts, metered repair feedback, corrected stop tokens, actual graded-output contrasts with full-input checks, practice-only `lab diagnose`, independent sampling and no-progress stops.

**Measured follow-up:** with identical initial prompts, unconstrained decoding produced **0 valid commands in 4 replies**; schema decoding produced **3 valid inventory calls in 3 replies**, then stopped as `stalled`. **Neither authored an edit or demonstrated a gain.** Real CLI/MCP and explicitly host-driven contrast/refusal checks passed separately. [Results, evidence and reproduction →](docs/EDITOR_EXPERIMENTS.md)

### Closed-loop discovery: test the mechanism before blaming the model

The new **opt-in discovery profile** adds all eight requested tools together: `hypothesize`, `inspect_gradients`, `compare_checkpoints`, `evaluate_candidate`, `rollback`, `analyze_failure`, `record_experiment`, `stop_and_explain`. Every attempted edit has a prospective hypothesis and an exact numerical audit; valid edits get actual paired practice feedback before revision. Gradient assistance is explicit, not disguised as no-gradient paper parity.

```bash
# A NEW scaffold, including the broader evaluation and its upstream license:
ouroboros init ouroboros-discovery --discovery
# Reuse the receipt-verified model from the quickstart above:
ouroboros lab tools --discovery
ouroboros lab experiment --model-dir ouroboros-lab/models/smollm2 \
  --data-dir ouroboros-discovery/discovery --workdir outputs/discovery-135m-01 --profile 135m
```

The fixed study has **240 examples / four categories**, including 60 independently authored GSM8K problems, **two seeds / eleven control-and-ablation arms**. No-edit, host-directed, matched-random and actual model-authored runs remain separate. Development seals before its first look, reserved independently confirms, final never reselects, and retention tests a fixed subsequent edit. A conservative **44-submission family correction**, strict accuracy gain and zero per-category forgetting stay in force. Lower loss or valid JSON cannot qualify as a useful autonomous edit.

**Measured status:** the complete **360M Instruct study was run and reviewed at both seeds: 22 arms, 64 model-editor replies, zero model-authored numerical edits, zero accepted candidates**. Host gradient controls made real changes but harmed practice and rolled back; they are not autonomous successes. The **135M study was OOM-killed after seven completed arms**; a smaller HOST dose improved practice but failed development (**p=0.5**). Its interrupted arm has no verified resident-restoration result and was not silently restarted. No gate was weakened or final result used to rescue a candidate. [Reviewed results and evidence →](docs/DISCOVERY_RESULTS.md) · [Protocol/reproduction →](docs/DISCOVERY_EXPERIMENTS.md)

**Still missing:** a validated beneficial autonomous pretrained edit unless its actual gates pass; learned improvement of future self-improvement; extended-training retention; a persistent isolated arbitrary-Python kernel; vLLM/tensor-parallel serving; vision/game/Go/diffusion adapters; and actual 27B/paper/full-standard-benchmark reproduction. **Full paper parity and capability gains are not claimed.**

### Research foundation: bounded experiments, not self-scored improvement

`ouroboros research` is a separate single-machine loop where the proposer, executor and evaluator are genuinely different processes. A researcher — a host reference or a real local model under a JSON schema — emits **versioned typed proposals** (competing hypotheses with uncertainty, alternatives, missing information, a numeric prediction and a falsification condition). The controller freezes the plan, budget and gate set before inference; workers train real LoRA/full-SFT/distillation candidates in fresh processes; one recipe is sealed by worst-seed practice accuracy; and validation, reserved, final, robustness and retention are each measured by a **fresh evaluator worker** at a conservative `0.05/44` family correction with zero per-domain forgetting.

```bash
ouroboros research tools      # versioned commands, authority and non-capabilities
ouroboros research deepseek   # what was adopted, refused or deferred from DeepSeek research
ouroboros research init ouroboros-research --model-path /path/to/local/model
ouroboros research run --config ouroboros-research/foundation.json --check-only
ouroboros research run --config ouroboros-research/foundation.json
ouroboros research verify-run --report ouroboros-research/runs/<run>/research.json
ouroboros research history --database ouroboros-research/experiments.sqlite3
```

Baselines are never overwritten, hidden prompts are consumed before inference and cannot be reopened, every artifact is write-once and hash-bound, and an append-only SQLite chain is the authoritative record. Model narratives are stored as **unverified claims**. There is no mid-step resume: an interrupted run needs a new plan and fresh confirmation.

**Measured status:** the complete gated loop has run only on a **randomly initialised ~1M-parameter GPT-2 fixture** with deliberately trivial constant-answer prompts, which proves plumbing and not capability. An earlier real-model attempt reached the follow-up stage and failed on a worker contract bug (it read a retention measurement it must not see); the fix makes the follow-up return `evaluation_requires_fresh_worker` and defers scoring to a separate process, and a clean rerun is still required. No pretrained gain, autonomous model discovery or improved future experiment selection is claimed. [Architecture, gates and subsystem status](docs/RESEARCH_FOUNDATION.md) · [DeepSeek applicability](papers/DEEPSEEK_APPLICABILITY.md)

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

Running experiments requires **both** `--allow-experiments` and an explicit `--config self_edit.json` or `--lab-config research_lab.json` allowlist at server startup. Research approval binds config, dataset and scripted-action bytes. There is no arbitrary shell, model download, patch-apply or delete tool. A subprocess timeout is not an OS security sandbox. [Tools, permissions and client setup →](docs/MCP.md)

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
| Same-model smoke | Constrained model-selected trials, strict measured gates, reversible accepted chains; still a finite menu |
| Research Lab | Frozen editor, model-generated closed-language programs, multi-tensor tools/controls, sealed checks, independent sessions and final test; optional schema-constrained commands and graded practice diagnostics |
| Discovery profile | Prospective hypotheses, actual iterative feedback, opt-in gradients, full attempted-edit ledger/replay, matched host/random controls, five-role broader data, final/retention probes; no guaranteed useful edit or learned meta-improvement |
| Research foundation | Separate researcher/executor/evaluator, versioned typed proposals, immutable plans, real LoRA/full SFT/distillation, per-seed clustered gates, robustness and four-cell retention, write-once artifacts, append-only SQLite memory, bounded watchdogs; complete loop only proven on a random fixture |
| Local SFT | Completion-masked Transformers/PEFT LoRA, replay anchors, original/latest retention and adapter rollback; GPU QLoRA untested |
| Weight tools | Bounded inspection, one-matrix trials plus atomic research programs and exact target-/full-state-bound sparse bundles |
| Agent access | JSON CLI, packaged `init`, dependency doctor, read-only-default stdio MCP, explicitly allowlisted execution |
| Other research primitives | Task arithmetic/TIES/DARE/SLERP, copied decoder candidates, explicit EasyEdit/mergekit bridges and review-only source proposals |
| Evidence | Measured predictions, fingerprints, failure reports, separate final test and CI; no invented benchmark gains |

[Training/data formats](docs/TRAINING.md) · [Weight-edit contract](docs/WEIGHT_EDITING.md) · [Integration boundaries](docs/INTEGRATIONS.md) · [Security](SECURITY.md)

## Research and real source checkouts

The [research index](papers/PAPERS_INDEX.md) contains **34 references**, including 12 recent primary-source reviews. Source-reported gains remain separate from local observations. The Astra/Qwen work is an **MIT CSAIL research blog**, not a Harvard paper or an Ouroboros reproduction.

Seven pinned submodules and five preserved snapshots are [inventoried](external/README.md), analyzed in the [repository deep dive](analysis/REPOSITORY_DEEP_DIVE.md), and labeled by actual integration level. Cloning a reference does not automatically install or execute it.

```bash
ouroboros repos sync    # explicit source fetch, not upstream execution
ouroboros repos verify
```

Core operation does not require those checkouts. The additional pretrained smoke source is pinned independently in [model provisioning code](ouroboros/model_files.py); only its model data/license are used. The optional command decoder was also actually cloned/read; its [separate package-source receipt](docs/evidence/structured-decoder-source.json) does not imply another vendored tree or upstream test run.

## Develop and verify

```bash
python scripts/install.py --profile train --dev
python -m pytest -q
ruff check ouroboros tests scripts
ruff format --check ouroboros tests scripts
python scripts/check_docs.py
# Explicit pretrained smoke; downloads only with --fetch:
python scripts/run_pretrained_smoke.py --workdir outputs/pretrained-smoke --fetch
# Reuses its verified model; actual and scripted research paths are labelled separately:
python scripts/run_research_smoke.py --workdir outputs/research-smoke --model-dir outputs/pretrained-smoke/models/smollm2
```

**Current local follow-up:** **463 tests passed**, plus **14 unittest checks**; `-m 'not ml'` returned **348 passed / 115 deselected** in this full environment (the earlier no-Torch counts, **291 passed / 24 expected optional skips / 111 ML deselected**, predate the research foundation and were not re-measured). The original discovery runtime passed **all nine push jobs and six ordinary PR jobs**; a separate read-only review recovered the complete 360M evidence. Its negative scientific outcome is not a CI failure or a test-fixture gain. Earlier 412-test distribution and 390-test editor snapshots remain separately labelled, not silently updated. [CI](.github/workflows/ci.yml) separates Python 3.10/3.11/3.12 core, offline ML/math/structured generation, MCP protocol and actual pretrained execution. See [testing](docs/TESTING.md) for exact source/run bindings, current-head status and distribution limits—not an assumption that an earlier green run validates later edits.

## Storage, releases, and license

Models, checkpoints, environments and run outputs are ignored by default. [Storage/LFS guidance](docs/STORAGE_AND_RELEASES.md) covers approved artifacts and safe run-scoped pruning. **No missing historical 600 MB payload has been recovered**, and the artifact registry remains empty; LFS patterns do not create uploads.

The existing [v0.3.0 pre-release](https://github.com/Mr-Nobody-Anonymous/ouroboros/releases/tag/v0.3.0) contains core source/checksums, not these new features, models or upstream trees. Historical releases remain unchanged. [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md) · [Agent guide](AGENTS.md)

Original code: [MIT](LICENSE). [Third-party attribution](THIRD_PARTY_NOTICES.md); models/sources retain their licenses. The README emblem is AI-generated artwork. Ouroboros is experimental software, not a guarantee of safety, generalization, or indefinite self-improvement.
