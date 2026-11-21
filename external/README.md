# External source inventory

Reviewed on **2026-11-19**. These are **12 source references**, not 12 automatically installed engines: seven pinned Git submodules and five preserved vendored snapshots. Five newly added sources complement the seven previously present references.

Use `ouroboros repos sync` to fetch locked submodules and `ouroboros repos verify` to check commit/content integrity. No upstream code is installed or executed by those commands. Minimal core training does not require any submodule. Normal GitHub source archives do not populate submodule contents; a Git clone plus initialization does.

| Source | Storage / pin | Integration level | License | Scope / limitation |
| --- | --- | --- | --- | --- |

| [EasyEdit](https://github.com/zjunlp/EasyEdit) | submodule `5c519afb14e3` | explicit-bridge | MIT | Localized editing; separate upstream environment and method hyperparameters required. |
| [mergekit](https://github.com/arcee-ai/mergekit) | submodule `0f1423ecf0f0` | recipe-and-executor | LGPL-3.0 | Validated recipe generation and opt-in subprocess execution; no bundled model weights. |
| [SEAL](https://github.com/Continual-Intelligence/SEAL) | vendored SHA-256 `c3444aa13452`; original revision unknown | research-reference | MIT | Existing snapshot; unknown original revision. Outer-loop SEAL RL is not executed by Ouroboros. |
| [peft](https://github.com/huggingface/peft) | vendored SHA-256 `78eaecf926bc`; original revision unknown | installed-package-backend | Apache-2.0 | Existing reference snapshot. Training uses the separately installed peft package, not a sys.path injection. |
| [trl](https://github.com/huggingface/trl) | vendored SHA-256 `240c3919c129`; original revision unknown | optional-preference-recipe | Apache-2.0 | Existing reference snapshot. Optional DPO recipe; GRPO not implemented in the default loop. |
| [repeng](https://github.com/vgel/repeng) | vendored SHA-256 `ec2ab2737dfa`; original revision unknown | research-reference | MIT | Existing snapshot; vector math is local. No automatic repeng hooks installed. |
| [self-instruct](https://github.com/yizhongw/self-instruct) | vendored SHA-256 `1419f4d1e61a`; original revision unknown | research-reference | Apache-2.0 | Existing partial source snapshot. Ouroboros supplies a verified generation callback boundary, not the original full training pipeline. |
| [weight-space-logs](https://github.com/RikiyaT/lm_understand-logs) | submodule `4d09ba6d6a7e` | research-logs | CUSTOM: see LICENSE | Cloned/read exp8 code/protocol; adapted owned reads/projections/controls/exact bundles, no upstream execution. Mixed licenses. |
| [WSP-Scope](https://github.com/WSP-Scope/WSP-Scope-Project) | submodule `d51670cab603` | paper-only | NOASSERTION | At this revision the checkout contains a PDF only, with no runnable code or license grant. Do not treat as an integrated engine. |
| [WeightBench](https://github.com/Aayush-Marasini/WeightBench) | submodule `2acd60172fc9` | external-benchmark | MIT | Cloned/inspected 42-task reference; no benchmark executed, sandbox/model/API setup separate. |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | submodule `d6de81643928` | opt-in-external-benchmark-driver | MIT | Cloned/inspected CLI; owned plan-first offline driver, actual harness benchmark unrun. Data/model setup separate. |
| [dgm](https://github.com/jennyzzt/dgm) | submodule `a565fd2d1dca` | research-reference | Apache-2.0 | Agent source-code evolution. Requires Docker, benchmark assets and API access; does not itself edit LLM weights. |

The complete lockfile is [sources.lock.json](sources.lock.json). Original upstream licenses remain in their checkouts/snapshots. No broad relicensing is implied. In particular the original WSP logs mix CC BY 4.0 data/artifacts, MIT scripts, GPL-3.0 quoted MinAtar source, and underlying benchmark licenses; inspect their LICENSE before redistribution. WSP-Scope has no license grant at this pin.

## Separately installed optional packages

The lockfile's `optional_installed_package_reviews` records **LM Format Enforcer 0.11.3**, MIT, actually cloned/read at `817f944fcc8851917c40300a47f62d1defc3ffc3`. Its [source receipt](../docs/evidence/structured-decoder-source.json) lists inspected files and owned tests. Ouroboros uses the separately installed package's Transformers prefix API with an owned action schema; no upstream code is copied into a thirteenth preserved source tree. `repos sync/verify` still cover only the twelve `sources` above, not an absent development cache or the installed wheel. Syntax filtering is not numeric/policy validation, a useful editing strategy or an upstream-suite result.

## Separately cloned discovery-data/model reviews

The lockfile's `development_data_model_source_reviews` and [byte-bound review receipt](../docs/evidence/discovery-sources.json) record two **actually cloned/read** development sources, not two more preserved submodules:

- **OpenAI grade-school-math** at `3101c7d5072418e28b9008a6636bde82a006892c`, MIT. Source README/license/train/test bytes reviewed and hashed. An owned adapter selects 60 disjoint question/numeric-reference pairs, preserves row hashes, excludes rationales and includes the MIT notice in the [five-role discovery data](../ouroboros/resources/discovery/manifest.json). No upstream Python executed, no performance-based selection, full benchmark or pretraining-decontamination claim.
- **Dadhichi-Sarker-Shayon/SmolLM2-360M-GGUF** at `a3f5ab2509573383064fd986bfff2d912dc4ba39`, Apache-2.0 assertion. README/license/recipe/converter reviewed for source and resource planning; no converter/weights executed. The clone contains metadata, not weights; declared payload hashes are unverified and refer to a **base** model. The independent larger-checkpoint CI instead explicitly provisions official **360M Instruct**, receipts a resolved revision before inference and never executes remote model code.

`repos sync/verify` intentionally still cover the twelve preserved `sources`, not absent development caches. Data adapter/output integrity has its own tests and manifest. A receipt of source review is not evidence that a model improved.

See [integration setup](../docs/INTEGRATIONS.md) and the [repository analysis](../analysis/REPOSITORY_DEEP_DIVE.md). Do not delete vendored data or rewrite unknown provenance merely to make a clone smaller. Source updates and deliberate patches should be separately reviewed and re-locked.
