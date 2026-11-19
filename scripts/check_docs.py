"""Check owned Markdown links, research metadata/index sync, and release-facing invariants offline."""

import json
import re
from datetime import date
from pathlib import Path
from urllib.parse import unquote, urlparse

if __package__:
    from .update_paper_index import render
else:
    from update_paper_index import render

ROOT = Path(__file__).resolve().parents[1]


def check(root=ROOT):
    root = Path(root).resolve()
    errors = []
    sources = json.loads((root / "external/sources.lock.json").read_text(encoding="utf-8"))[
        "sources"
    ]
    # Source-only archives omit upstream trees; a normal clone may omit submodule contents.
    # Only absent/empty locked trees get this exemption, never a missing owned/local file.
    unavailable_sources = [
        source["path"]
        for source in sources
        if not (root / source["path"]).exists() or not any((root / source["path"]).iterdir())
    ]

    def unavailable_reference(relative):
        return any(
            relative == path or relative.startswith(path + "/") for path in unavailable_sources
        )

    files = list(root.glob("*.md"))
    for directory in ("docs", "analysis", "papers", "sandbox"):
        files.extend((root / directory).glob("*.md"))
    files.append(root / "external/README.md")
    for file in files:
        for raw in re.findall(r"\[[^\]]*\]\(([^)]+)\)", file.read_text(encoding="utf-8")):
            target = raw.split(' "', 1)[0]
            parsed = urlparse(target)
            if parsed.scheme or target.startswith("#"):
                continue
            path = (file.parent / unquote(parsed.path)).resolve()
            if not path.is_relative_to(root):
                errors.append(f"{file.relative_to(root)}: link escapes repository: {target}")
            elif not path.exists():
                relative = str(path.relative_to(root))
                if not unavailable_reference(relative):
                    errors.append(f"{file.relative_to(root)}: missing link: {target}")
    catalogue = json.loads((root / "papers/research.json").read_text(encoding="utf-8"))
    papers = catalogue["papers"]
    if len({paper["id"] for paper in papers}) != len(papers):
        errors.append("Duplicate research IDs")
    as_of = date.fromisoformat(catalogue["as_of"])
    for paper in papers:
        if date.fromisoformat(paper["published"]) > as_of:
            errors.append(f"Future paper: {paper['id']}")
        if not paper["url"].startswith("https://") or (
            not (root / paper["local_path"]).exists()
            and not unavailable_reference(paper["local_path"])
        ):
            errors.append(f"Invalid research source or local mapping: {paper['id']}")
        if paper["reproduced_here"]:
            errors.append(
                f"A paper was marked reproduced without benchmark evidence: {paper['id']}"
            )
    if (root / "papers/PAPERS_INDEX.md").read_text(encoding="utf-8") != render(root):
        errors.append("Paper index out of sync; run python scripts/update_paper_index.py")
    if errors:
        raise ValueError("\n".join(errors))
    return {
        "owned_markdown_files": len(files),
        "research_references": len(papers),
        "unavailable_upstream_trees": unavailable_sources,
        "status": "ok",
    }


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
