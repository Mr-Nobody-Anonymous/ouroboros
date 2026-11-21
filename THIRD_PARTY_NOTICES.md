# Third-party sources, data and attribution

Ouroboros's original code is [MIT](LICENSE). That license does **not** relicense upstream repositories, model weights, benchmark questions, experiment logs or dependencies. The pinned [source inventory](external/sources.lock.json) records each reference's origin/license/integration level. Referenced submodule trees are excluded from core source archives and are not bundled as Ouroboros-authored code.

## Weight-space programming protocols and algebra

The research tools are owned implementations informed by **Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab, MIT CSAIL**, [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (November 2, 2026), and its released [experiment logs/code](https://github.com/RikiyaT/lm_understand-logs), reviewed at `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`. In particular, experiment 8's projection/column selection, final-box grading, separate evaluation and exact edit reconstruction informed the contracts of `ouroboros/lab/`. No benchmark questions, experiment logs or network payloads from that WSP repository are copied into the owned package. The separately licensed GSM8K selection is attributed below. No upstream agent harness was executed.

At that revision, upstream **author-written Python/shell scripts** are MIT; **logs/data/networks** are CC BY 4.0 except third-party data; the quoted **MinAtar source** is GPL-3.0. These scopes must not be collapsed into a single MIT grant. Dataset content remains under its original license. The following MIT notice is retained for the script-informed adaptation:

> MIT License
>
> Copyright (c) 2026 the authors of "LLMs can understand and write neural network weights"
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Self-rewriting methods

The frozen-editor/target distinction, paired metered practice, independent sessions, reserved submit/withdraw check and problem-clustered evaluation are owned implementations informed by **Jiajun Fan et al.**, [Language Models Can Rewrite Their Own Weights, v2](https://arxiv.org/abs/2611.21885v2), November 18, 2026. This is a methods attribution, not a claim to have cloned author code or reproduced the paper. The [parity audit](docs/PAPER_PARITY.md) identifies deliberate differences, unimplemented surfaces and unrun experiments.

## Libraries and independently installed harnesses

The optional `boxed_math` grader calls installed **Math-Verify 0.9.0** (Hugging Face, Apache-2.0) through its public parse/verify API; no library code is copied. Its transitive parsers/SymPy retain their respective licenses. No unsafe expression `eval` or copied Hendrycks normalizer is used in Ouroboros's own grader. Installing a dependency grants no rights to benchmark content.

The external evaluation driver was checked against EleutherAI's **lm-evaluation-harness** CLI at `d6de81643928d653435c431bae19945d41d32520` (MIT). It invokes a **separately installed trusted executable** only with explicit permission; it does not vendor, automatically install or certify that executable's revision, tasks or datasets. **WeightBench** at `2acd60172fc9758eb9030b1cab4a2ff29be5c993` is a cloned/inspected reference, not an executed benchmark integration. See [integration boundaries](docs/INTEGRATIONS.md).

## Optional structured-command decoder

The `structured` extra calls **Noam Gat's LM Format Enforcer 0.11.3** ([source](https://github.com/noamgat/lm-format-enforcer), MIT, Copyright (c) 2023 Noam Gat). Its source was actually cloned/reviewed at `817f944fcc8851917c40300a47f62d1defc3ffc3`: JSON schema/parser configuration, token enforcement/caching and the Transformers prefix integration. Ouroboros adapts that public API with an owned closed action schema, public-state-bound enums, cached tokenizer data, model-EOS alignment and independent host validation. No library code is copied/vendored; installed dependencies and their transitive packages retain their licenses. [Review receipt](docs/evidence/structured-decoder-source.json), [usage and limitations](docs/EDITOR_EXPERIMENTS.md). Grammar filtering is not an edit-quality or safety guarantee.

## Pretrained smoke model

The optional smoke provisioner clones [simonw/llm-smollm2](https://github.com/simonw/llm-smollm2) at `afed0963b3a8bcca17c83f451a161eb2edbc59fe`, reads its known hash/size model data and license, and never executes its plugin. The data originates from **HuggingFaceTB/SmolLM2-135M-Instruct**, redistributed as a **QuantFactory Q4_1 GGUF** and dequantized into float32 safetensors. Original precision is not recovered. The model's Apache-2.0 notice/license, source attribution, conversion and output hashes are recorded with the separately provisioned receipt. No model weights are distributed in this source repository/release.

The **legacy** tiny public examples and synthetic text contrasts are owned plumbing fixtures, not copies of MATH/GSM8K/IFEval/DeepScaleR or independently certified labels. The new opt-in discovery dataset separately includes the explicitly attributed GSM8K selection below. Importing local user datasets records **asserted** source/license metadata, not a legal/correctness attestation. Always review upstream/model/dataset terms before redistribution.

## Discovery data: selected GSM8K problems

The five `ouroboros/resources/discovery/*.json` role files contain **60 selected question/numeric-reference pairs** adapted from OpenAI's [grade-school-math](https://github.com/openai/grade-school-math), actually cloned/read at `3101c7d5072418e28b9008a6636bde82a006892c`, plus 180 owned synthetic diagnostic examples. Upstream source code was not executed. Numeric answers are extracted after `####` and commas removed; chain-of-thought rationales are not distributed as editor inputs. The [manifest](ouroboros/resources/discovery/manifest.json) preserves upstream train/test paths, raw-row hashes/indices, selection policy, source-byte hashes and all output hashes.

**Copyright (c) 2021 OpenAI. MIT License.** The full required permission/warranty notice is retained in [GSM8K_LICENSE.txt](ouroboros/resources/discovery/GSM8K_LICENSE.txt) and included in package/scaffold distributions. These are selected diagnostics, **not** full GSM8K evaluation, independently certified labels or proof of pretraining decontamination. The OpenAI license does not certify our selection protocol or scientific conclusions.

## Larger checkpoint source review

[Dadhichi-Sarker-Shayon/SmolLM2-360M-GGUF](https://github.com/Dadhichi-Sarker-Shayon/SmolLM2-360M-GGUF) was actually cloned/read at `a3f5ab2509573383064fd986bfff2d912dc4ba39`. Its README/license/conversion recipe were reviewed; no converter code or model payload was executed/provisioned from it. Declared upstream weights are **base**, not Instruct, and declared payload hashes are not independently verified locally. The configured GitHub experiment instead explicitly provisions the official **HuggingFaceTB/SmolLM2-360M-Instruct**, receipts its resolved revision/bytes, and loads only data files. No larger weights are included in this source package; consult the official checkpoint's license/receipt before redistribution.
