# ROME and MEMIT: factual editing is more than rank-one algebra

Primary references:

- Meng et al., [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262), 2022 (ROME).
- Meng et al., [Mass-Editing Memory in a Transformer](https://arxiv.org/abs/2210.07229), 2022 (MEMIT).

ROME combines localization/causal tracing, key/value construction and covariance-aware changes to selected transformer weights. MEMIT distributes many memory edits across layers. A simplified covariance-aware rank-one update can be written

`delta_W = (v - W k) (k^T C^-1) / (k^T C^-1 k)`.

This does not by itself specify the complete algorithm: subject token selection, value optimization, layer choice, covariance estimation, model orientation and evaluation are essential. Factual editing can damage locality or broader capabilities; exact success on one edited prompt is insufficient.

## Local status

`WeightEditor.surgical_rank_one_edit` is a **regularized least-squares reference primitive** using a supplied key/value, not a full ROME or MEMIT reproduction. Shape/value checks and norm limits are tested, but those tests do not measure factual editing on an LLM.

`EasyEditBridge` explicitly invokes the installed upstream editor with a reviewed method-specific config or raises. The `external/EasyEdit` gitlink is now accompanied by `.gitmodules` and a locked revision. No bridge reports “delta applied” merely because an empty source directory exists. See [integration setup](../docs/INTEGRATIONS.md) for dependency/statistics/remote-code caveats and independent locality/portability evaluation requirements.
