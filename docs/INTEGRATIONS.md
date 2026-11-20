# Upstream integration boundaries

Cloning source is not installation, runtime integration, reproduced results or a security review. See the [locked inventory](../external/sources.lock.json) for exact revisions/snapshot hashes and [comparative analysis](../analysis/REPOSITORY_DEEP_DIVE.md) for selection rationale. No upstream files were silently patched; integration behavior is implemented in the owned `ouroboros/` code.

## Acquire and verify

```bash
ouroboros repos list
ouroboros repos sync
ouroboros repos verify
# Optional single source fetch:
ouroboros repos sync --name WeightBench
```

`sync` only initializes committed submodules, never follows a floating remote branch, installs dependencies or runs upstream code. `GIT_LFS_SKIP_SMUDGE=1` is set during fetch to avoid implicitly pulling large upstream objects. The five existing vendored source snapshots remain as they were. Their original commit revisions were not recorded, so we hash their tracked contents and mark the revisions unknown instead of inventing provenance.

## PEFT / Transformers: real training backend

The [local training guide](TRAINING.md) describes the installed-package backend and tested dependency combination. Vendored PEFT is a reference copy; editing it does not change the installed PEFT wheel. Install a reviewed fork explicitly in a separate environment if you need to change the upstream implementation.

The owned `LocalCausalLM` loader also supports explicit direct-weight trials without inserting LoRA. It freezes a local safetensors base, records base/tokenizer hashes and shares SFT's prompt/greedy inference path. PEFT is only imported by the SFT subclass, though the `train` convenience extra installs both.

## Numeric inspection and direct trials: owned tools, not an upstream reproduction

[Weight-edit commands/APIs](WEIGHT_EDITING.md) provide bounded read-only inspection without Torch, measured one-matrix trials with unconditional restoration, accepted-only sparse data exports and explicit exact apply/reverse. The original WSP logs motivated their contracts; neither those tools nor a passing 2×2 fixture demonstrate reproduction of Astra, WSP-Scope or WeightBench results. Callers still choose the model, target, direction and independent evaluation. Keep direct base edits outside adapter-only SFT snapshots/exports unless you implement/test their backend state contract.

## Same-model selection, pretrained smoke source and MCP

The owned [self-edit controller](SELF_EDITING.md) adds constrained next-token menu selection using the **same** target model, not an external autonomous agent or the upstream self-rewriting algorithm. It composes existing measured trials, original/latest gates and exact accepted-patch reversal without mixing base edits into adapter-only SFT. The real 135M pretrained smoke rejected two nonzero edits; no capability gain is claimed.

The independently pinned `simonw/llm-smollm2` distribution was actually cloned and its README/license/data layout inspected. Only its known-size/SHA-256 pretrained Q4_1 GGUF and license are adapted through installed Transformers/GGUF conversion; the cloned plugin is never imported, installed or executed. Source commit, original model/quantizer attribution, conversion caveat and output hashes are recorded in the model receipt. This optional downloaded model data is **not** a new vendored source tree or an expansion of the twelve-source integration inventory. `models fetch` is an alternative explicit Hub provisioning path; normal inference remains local-only.

The optional official Python MCP SDK powers a [stdio server](MCP.md). Read-only tools are default; explicit startup opt-in plus a config allowlist enables fixed-command offline-Hub child experiments. No cloud-app connector, HTTP service, shell tool or arbitrary generated-code executor is implied. Ordinary tests exercise actual MCP protocol initialization/tool calls without model downloads; pretrained smoke is a separately opted-in workflow.

## Paper-inspired research workflow: adapted owned code

The [Research Lab](RESEARCH_LAB.md) unifies the frozen-editor/target protocol, bounded model-authored JSON numerical programs, actual hidden/MLP/logprob/attention reads, multi-layer edits, norm-matched/no-op/layer controls, dose sweeps, saved alternatives, reserved seal/confirm/withdraw, independent sessions and exact full-state-bound bundles. Local dataset imports and a plan-first external benchmark driver are explicit. This is **not full paper parity**: see every feature/disposition in [the audit](PAPER_PARITY.md).

Three relevant trees were actually initialized at their locked commits in this phase: original WSP logs, WeightBench and lm-evaluation-harness. The experiment 8 scripts/protocol/license and harness CLI were inspected; their projection/intervention, final-box grading and separate-evaluation contracts informed owned modules, with [attribution/license scope](../THIRD_PARTY_NOTICES.md). No author script, upstream agent, paid provider or full benchmark was executed. Four other submodules may be absent; no new 12/12 verification is implied.

The real Q4_1-derived 135M model produced **zero valid research actions in two replies**. The separately scripted 16-action integration changed 1,152 entries across two tensors, scored 0→0/rejected, restored. Real CLI and official-SDK stdio MCP execution/readback also succeeded as plumbing; no autonomous circuit discovery, benchmark gain, 27B/vision experiment or arbitrary-Python sandbox is claimed.

## EasyEdit: explicit call, separate environment

The bridge supports `ROME`, `MEMIT`, `MEND` and `FT` with an installed compatible `easyeditor` package and a real method-specific hyperparameter YAML.

```python
from ouroboros.integrations.easyedit_bridge import EasyEditBridge

result = EasyEditBridge("ROME").execute_direct_edit(
    model_name="/path/to/a/model-supported-by-EasyEdit",
    target_fact={"prompt": "The Eiffel Tower is located in", "subject": "Eiffel Tower", "target_new": "Paris"},
    hparams="/path/to/reviewed/rome_hparams.yaml",
)
# result holds actual metrics, edited_model and original_weights.
# status='executed' is not validated_improvement=True; run independent tests.
```

Missing installation, invalid config or upstream failure raises an error; there is no fabricated success or silent simulated fallback. The bridge does not add an empty external directory to `sys.path` and call it available.

**Caution:** upstream EasyEdit may fetch covariance statistics, model/tokenizer files or use remote model code according to its own loader/config. Those are not covered by core Ouroboros's offline/no-remote-code guarantee. Provision artifacts, inspect its implementation and isolate the environment. Large factual-editing jobs and algorithm-specific locality/portability benchmarks were not executed here.

## mergekit: recipes and explicit execution

```python
from ouroboros.integrations.mergekit_bridge import MergeKitBridge

bridge = MergeKitBridge()
recipe = bridge.generate_slerp_recipe("/models/a", "/models/b", "outputs/merge.yml", interpolation_t=0.5)
# Creation above only writes a recipe. The following is an explicit expensive operation:
# bridge.execute(recipe, "outputs/merged-model", allow_execution=True, timeout=300)
```

SLERP recipes use whole-model sources rather than hardcoded 32-layer ranges. DARE-TIES requires nonempty experts and a valid density. Execution needs `mergekit-yaml` on PATH, uses argument arrays rather than a shell, refuses existing output, and propagates failures/timeouts. Review tokenizer/base-architecture compatibility and memory/storage requirements. Model identifiers in an upstream recipe may trigger upstream downloads; use local paths for offline work.

## TRL / SEAL / self-instruct / repeng

These are preserved source references, not a single universally compatible installed ecosystem. Ouroboros provides verified data/judge interfaces, vector utilities and local SFT. It does **not** silently train DPO/GRPO, run SEAL's outer RL algorithm, synthesize a complete Self-Instruct corpus or install repeng runtime hooks. Follow the pinned upstream instructions for those experiments and keep independent evaluation/rollback around any resulting checkpoint.

## Independent evaluation

[EleutherAI's harness](../external/lm-evaluation-harness/README.md) supports standard tasks and PEFT adapters. Install it in a separate environment following its README. Provision the corresponding licensed datasets first; normal upstream invocation can otherwise download them. Its evolving CLI is not invoked automatically by the core loop. The explicit `ouroboros lab benchmark` bridge returns a JSON plan by default; `--execute` runs fixed greedy `lm-eval run` arguments with offline flags, new outputs, logs and timeout cleanup. The command boundary was tested with a labelled fixture executable, **not a real installed harness/benchmark run**. It does not pin the executable on PATH, sandbox upstream task code, apply bundles or implement paper sampling parity. Consult `lm-eval run -h` / the reviewed revision before use.

[WeightBench](../external/WeightBench/README.md) supplies an OS-level harness and task registry for weight-space programming. It requires separate setup, sandbox checks, and sometimes external agent/model access. Its 42-task benchmark results are not the same as Ouroboros's unit tests. No WeightBench benchmark run or paid model call was performed in this session.

## Research logs, WSP-Scope and DGM

- [Original weight-space logs](../external/weight-space-logs/README.md) include protocols, scripts, prompts, artifacts and evaluation evidence. They have mixed CC BY/MIT/GPL and underlying dataset licenses. They do not implement a generally reliable local autonomous tuner by themselves.
- `external/WSP-Scope` is pinned for paper provenance. At the reviewed revision it contains a PDF only, without code or a license grant. It must not be described as an integrated runnable framework or bundled under Ouroboros's MIT license.
- [Darwin Gödel Machine](../external/dgm/README.md) evolves coding-agent source; it needs Docker, task assets and API access. It is not evidence of direct model-weight self-rewriting. We reference it for source-proposal workflows, never execute its autonomous loop automatically.
