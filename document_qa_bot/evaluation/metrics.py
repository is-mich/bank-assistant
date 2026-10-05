"""
evaluation/metrics.py
---------------------
Retrieval metrics (recall@k, MRR) and cost, independent of any LLM or vector store.

Gold relevance comes from the `source_doc` column of the question set:
  "file.pdf"              any chunk of that document is relevant
  "file.pdf:39"           only chunks from page 39 (1-based, as shown in the UI)
  "file.pdf:39;file.pdf:42"  several references, joined with ';'
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class GoldRef:
    """A relevant location: a document, optionally restricted to one 1-based page."""
    source: str
    page: Optional[int] = None


def parse_gold(source_doc: str) -> list[GoldRef]:
    """Parse the `source_doc` cell of the question set into gold references."""
    refs = []
    for part in (source_doc or "").split(";"):
        part = part.strip()
        if not part:
            continue
        name, sep, page = part.rpartition(":")
        if sep and page.strip().isdigit():
            refs.append(GoldRef(name.strip(), int(page)))
        else:
            refs.append(GoldRef(part))
    return refs


def chunk_matches(chunk: dict, ref: GoldRef) -> bool:
    """True if a retrieved chunk ({"source", "page"} with 0-based page) satisfies a gold reference."""
    if Path(str(chunk.get("source", ""))).name != Path(ref.source).name:
        return False
    if ref.page is None:
        return True
    page = chunk.get("page")
    return page not in ("", None) and int(page) + 1 == ref.page


def recall_at_k(retrieved: list[dict], gold: list[GoldRef], k: int) -> Optional[float]:
    """Share of gold references hit by at least one of the top-k chunks; None without gold."""
    if not gold:
        return None
    top = retrieved[:k]
    hits = sum(1 for ref in gold if any(chunk_matches(c, ref) for c in top))
    return hits / len(gold)


def reciprocal_rank(retrieved: list[dict], gold: list[GoldRef], k: int) -> Optional[float]:
    """1 / rank of the first top-k chunk matching any gold reference (0 if none); None without gold."""
    if not gold:
        return None
    for rank, chunk in enumerate(retrieved[:k], start=1):
        if any(chunk_matches(chunk, ref) for ref in gold):
            return 1.0 / rank
    return 0.0


def cost_usd(prompt_tokens: int, completion_tokens: int,
             price_in_per_1m: float, price_out_per_1m: float) -> float:
    """Cost of one question from real token usage; prices are USD per 1M tokens."""
    return (prompt_tokens * price_in_per_1m + completion_tokens * price_out_per_1m) / 1_000_000
