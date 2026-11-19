# Audit, corrections and migration from the original demonstration

Audit baseline: repository commit `97181d38a12438c1a5c9db753a3552e5edf48539`, November 19, 2026. The historical transcript described more than could be verified. This document separates observed state, corrections and implemented replacements.

## Evidence-based corrections

| Earlier assertion | Observed state / correction | Change |
| --- | --- | --- |
| Complete self-fine-tuning cycle with improving math scores | Loop trained random matrices using unrelated random gradients and invented score boosts | Actual NumPy reference gradients and optional local causal-LM LoRA; measured predictions only |
| Rollback after degradation | Only score variables were retained; arrays were not restored | Snapshot/restore of actual adapter tensors and RNG state |
| Saved model/adapters | Checkpoints wrote configuration metadata only | Real NumPy/safetensors adapter exports, manifests, reload tests |
| EasyEdit applied a delta | Bridge returned success without an editor call | Explicit real invocation or actionable failure; no silent simulation |
| Seven initialized submodules | Five vendored snapshots plus two bare gitlinks; `.gitmodules` absent | Preserve five snapshots; repair two pins; add five targeted pinned sources |
| Pareto storage pruning | Sorted mean accuracy, never inspected sizes, eligible unknown directories | Actual accuracy/bytes frontier, ownership/checksum/name/symlink checks and preview |
| Sparse top-k protects all other parameters | Threshold ties could select more than k; zero ratio ignored | Exact index budget, floor/zero support and explicit behavioral caveats |
| SVD diagnoses a repetition circuit / reproduces Astra | Dominant singular direction alone is not causal evidence; computed edits were not applied to a real LLM | Rename/deprecate misleading method, contrastive activation API, explicit reversible candidate edits |
| Self-rewarding and STaR | Style-word scores, canned candidates, random rollouts; high confidence could accept a wrong target | Explicit judges/verifiers; abstention and exact canonical correctness curation |
| Syntax checks provide source sandbox | Python parsing/compilation does not isolate code | No auto-execution; bounded arithmetic interpreter and separate container policy |
| Replay eliminates collapse; universal 20–30% anchor theorem | Not established by cited work; ratio was computed against new samples rather than final batch | Configurable correctly calculated replay, verified source records, no guarantee |
| arXiv:2404.14818 is MoE-CL with zero forgetting | [Actual paper](https://arxiv.org/abs/2404.14818) is **Simplicial volume of manifolds fibering with connected structure group** | Remove incorrect AI attribution and guarantee |
| Harvard/Astra peer-reviewed paper | [MIT CSAIL research blog](https://rikiyat.github.io/blog/weight-space-programming/), November 2; no formal publication/DOI listed | Exact source/type, reported +40.2 percentage points, no local replication claim |
| TIES and DARE share one paper | [TIES 2306.01708](https://arxiv.org/abs/2306.01708) and [DARE 2311.03099](https://arxiv.org/abs/2311.03099) are distinct | Correct bibliography and implementation mapping |
| SIA is an unspecific 2025 source | [Exact preprint 2605.27276](https://arxiv.org/abs/2605.27276), May 2026 | Replace homepage link and date |
| LFS uploaded deleted 600+ MB and release included it | No pointers/recoverable payload found; `gh release view v0.1.0 --json assets` returned an empty list | Empty honest artifact inventory, registration/verification tooling, no fabricated upload |
| A normal LFS clone gets pointers only | Normally an installed LFS smudge filter downloads objects during checkout | Document `GIT_LFS_SKIP_SMUDGE=1`, then selective `git lfs pull` |

## Preserved and deliberately not changed

- The repository is already named `ouroboros`; its root/remote name was not renamed again.
- Existing vendored SEAL/PEFT/TRL/repeng/self-instruct source/data were preserved, not purged as disposable caches. Unknown original revisions are recorded as unknown with tracked-content SHA-256.
- Existing release history was not rewritten as if older claims had always been correct.
- No absent model weights, benchmark datasets, credentials or external private files were synthesized or purportedly recovered.
- No unreviewed upstream code was installed automatically or given host credentials. No paid model/API or large public benchmark run was executed.
- All implementation work stays on the assigned current branch; no hidden experiment branches or automatic merges.

## Migration

1. Install the minimal package with `pip install -e '.[dev]'`. Heavy LLM training is an explicit `train` extra.
2. Run `ouroboros demo` for measured plumbing tests. Old `--generations/--domain/--simulate` invocations warn and route to that demo; they are not math evaluations.
3. For real models, supply local safetensors/tokenizer files and separate training/validation/replay data via `examples/local_sft.json`.
4. Old metadata-only adapter directories are **not valid new checkpoints**. Re-run a real training/export or import real tensors with verified provenance; no tensor content can be reconstructed from a configuration JSON.
5. `SelfEvaluator` no longer supplies fabricated baseline numbers. Provide measured scores/predictions; exact match replaces permissive substring matching.
6. Judges, generators and verifiers must be explicit. Fake success/fallback APIs now raise instead of pretending to complete work. Tied preferences abstain.
7. Pruning requires a run ID and managed ownership/checksums; old arbitrary folders are left alone. The CLI previews unless `--apply` is supplied.
8. Run `ouroboros repos sync`/`verify` for the seven pinned submodules and five preserved source snapshots. Read the integration status before treating a clone as executable functionality.
9. Use the artifact register/verify tools only for files actually present and licensed. Git LFS installation was not completed in this sandbox: a release-asset download failed with `EOF`, and the exact cause was not established. Source attributes/scripts are prepared; no LFS upload is claimed.

See [training](TRAINING.md), [storage/releases](STORAGE_AND_RELEASES.md), and [test evidence](TESTING.md). The research and engineering infrastructure is substantially more honest and usable, but it is not a substitute for independent scientific evaluation.
