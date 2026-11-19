"""Regenerate the human index from the curated, source-linked research catalogue (no network)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def render(root=ROOT):
    catalogue = json.loads((Path(root) / "papers/research.json").read_text(encoding="utf-8"))
    papers = catalogue["papers"]
    recent = [
        p
        for p in papers
        if catalogue["recent_window"]["start"] <= p["published"] <= catalogue["as_of"]
    ]
    lines = [
        "# Research index: model adaptation, weight editing, and self-evolution",
        "",
        f"As of **{catalogue['as_of']}**. {len(papers)} references; {len(recent)} recent primary-source reviews in the **{catalogue['recent_window']['start']} to {catalogue['as_of']}** window.",
        "",
        "This is a curated selection, not an exhaustive survey. Blog posts, preprints and journal papers are labeled separately. Source-reported results have **not** been reproduced by Ouroboros. An implementation link is a partial mapping unless explicitly stated otherwise.",
        "",
        "Machine-readable [catalogue](research.json), abbreviated-author [BibTeX](references.bib), [recent evidence review](RECENT_RESEARCH_2026.md), [architecture](../docs/SELF_EVOLVING_ARCHITECTURE.md), and [source inventory](../external/README.md).",
        "",
        "## Recent work (first release inside the requested three-month window)",
        "",
        "| First released | Primary source / type | Relevance and limitations | Local status |",
        "| --- | --- | --- | --- |",
    ]
    for p in recent:
        lines.append(
            f"| {p['published']} | [{p['title']}]({p['url']}) — {p['kind']} | {p['summary']} | [{p['implementation_status']}](../{p['local_path']}) |"
        )
    lines += [
        "",
        "## Foundations and earlier related work",
        "",
        "| Primary reference | First released | Mechanism / caveat | Local status |",
        "| --- | --- | --- | --- |",
    ]
    for p in papers:
        if p in recent:
            continue
        lines.append(
            f"| [{p['title']}]({p['url']}) | {p['published']} | {p['summary']} | [{p['implementation_status']}](../{p['local_path']}) |"
        )
    lines += [
        "",
        "## Corrections and reading policy",
        "",
        "- The Astra/Qwen source is an MIT CSAIL research blog post, not a Harvard arXiv paper. The reported 7.9% to 48.1% change is +40.2 percentage points, not a local result or a guarantee.",
        "- arXiv:2404.14818 was previously mislabeled as MoE-CL. It is an unrelated topology paper and has been removed from the AI research catalogue; see the [audit](../docs/AUDIT_AND_MIGRATION.md).",
        "- TIES (2306.01708) and DARE (2311.03099) are distinct papers. Rank-one least-squares utilities are not complete ROME/MEMIT implementations.",
        "- SIA now links to its exact May 2026 preprint, not an unspecific homepage or an invented 2025 citation.",
        "- Replay mitigates drift; neither a universal 20–30% ratio nor guaranteed zero forgetting is established here.",
        "- Recent entries record primary abstracts/selected methods actually consulted on November 19. Foundational entries are background references, not a claim that every full paper was newly re-read.",
        "- Original summaries and bibliographic links are stored here; full papers are not automatically relicensed or redistributed. Upstream WSP-Scope is paper-only at the locked revision.",
        "",
    ]
    return "\n".join(lines)


def main():
    (ROOT / "papers/PAPERS_INDEX.md").write_text(render(), encoding="utf-8")


if __name__ == "__main__":
    main()
