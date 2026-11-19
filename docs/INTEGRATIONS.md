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

[EleutherAI's harness](../external/lm-evaluation-harness/README.md) supports standard tasks and PEFT adapters. Install it in a separate environment following its README. Provision the corresponding licensed datasets first; normal upstream invocation can otherwise download them. Its evolving CLI is not invoked automatically by the core loop. Consult `lm-eval run -h` / the pinned CLI guide rather than assuming old flags stay unchanged.

[WeightBench](../external/WeightBench/README.md) supplies an OS-level harness and task registry for weight-space programming. It requires separate setup, sandbox checks, and sometimes external agent/model access. Its 42-task benchmark results are not the same as Ouroboros's unit tests. No WeightBench benchmark run or paid model call was performed in this session.

## Research logs, WSP-Scope and DGM

- [Original weight-space logs](../external/weight-space-logs/README.md) include protocols, scripts, prompts, artifacts and evaluation evidence. They have mixed CC BY/MIT/GPL and underlying dataset licenses. They do not implement a generally reliable local autonomous tuner by themselves.
- `external/WSP-Scope` is pinned for paper provenance. At the reviewed revision it contains a PDF only, without code or a license grant. It must not be described as an integrated runnable framework or bundled under Ouroboros's MIT license.
- [Darwin Gödel Machine](../external/dgm/README.md) evolves coding-agent source; it needs Docker, task assets and API access. It is not evidence of direct model-weight self-rewriting. We reference it for source-proposal workflows, never execute its autonomous loop automatically.
