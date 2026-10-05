"""
ingestion/structure_splitter.py
-------------------------------
Splits legal/contractual text along its own structure instead of fixed character windows.

Written for Greek general-terms documents: `ΚΕΦΑΛΑΙΟ n`, `ΕΝΟΤΗΤΑ I`, numbered clauses
(`1.`) and sub-clauses (`2.2.`). Other documents still work; they just have fewer headings.

Rules:
  * a chunk never crosses a page boundary, so "σελ. N" citations stay exact;
  * chunk boundaries fall on clause / sub-clause starts; small clauses are packed together;
  * a clause longer than the limit is split with the recursive splitter;
  * every chunk starts with a header giving its chapter and section (and the clause title
    when it continues a clause), so it is understandable on its own;
  * table-of-contents pages and bare page-number lines are dropped.
"""

import re
from dataclasses import dataclass

# pyrefly: ignore [missing-import]
from langchain_core.documents import Document
# pyrefly: ignore [missing-import]
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHAPTER_RE = re.compile(r"^\s*ΚΕΦΑΛΑΙΟ\s+\d+")
SECTION_RE = re.compile(r"^\s*ΕΝΟΤΗΤΑ\s+[IVXΙΧ]+")  # Latin and Greek capital I and X
CLAUSE_RE = re.compile(r"^\s*\d{1,2}\.\s+\S")
SUBCLAUSE_RE = re.compile(r"^\s*\d{1,2}\.\d{1,2}\.\s+\S")
TOC_LINE_RE = re.compile(r"(\x08|\.{3,}|\t)\s*\d+\s*$")
PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,3}\s*$")

TOC_MIN_LINES = 3          # a page with this many "title ..... 12" lines is a table of contents
CLAUSE_TITLE_CHARS = 80
MIN_BODY_CHARS = 200       # lower bound for the text budget of a chunk after its header
SEPARATORS = ["\n\n", "\n", ". ", "; ", " "]


@dataclass
class _Unit:
    """A run of lines that starts at a structural boundary and stays on one page."""
    text: str
    metadata: dict
    chapter: str
    section: str
    clause: str
    continued: bool  # True if the text does not start at a heading or (sub-)clause


def _squash(line: str) -> str:
    # U+FFFE is how the PDF font encodes some hyphens ("ΚΑΤΑΘΕΣΕΙΣ-ΕΝΤΟΛΕΣ").
    line = line.replace("\ufffe", "-")
    return " ".join(re.sub(r"[\x00-\x1f]", " ", line).split())


def _header(chapter: str, section: str, clause: str, continued: bool) -> str:
    parts = [p for p in (chapter, section) if p]
    if continued and clause:
        parts.append(f"{clause} (συνέχεια)")
    return " | ".join(parts)


def _clean_pages(pages: list[Document]) -> list[tuple[Document, list[str]]]:
    """Drop table-of-contents pages and the page-number line; return (page, lines) pairs."""
    kept = []
    for doc in pages:
        lines = [line.rstrip() for line in doc.page_content.splitlines()]
        if sum(1 for line in lines if TOC_LINE_RE.search(line)) >= TOC_MIN_LINES:
            continue
        if lines and PAGE_NUMBER_RE.match(lines[0]):
            lines = lines[1:]
        kept.append((doc, [line for line in lines if line.strip()]))
    return kept


def _build_units(pages_lines: list[tuple[Document, list[str]]]) -> list[_Unit]:
    units: list[_Unit] = []
    chapter = section = clause = ""
    for doc, lines in pages_lines:
        buf: list[str] = []
        starts = False  # a page opens mid-sentence unless its first line is a boundary

        def flush() -> None:
            text = "\n".join(buf).strip()
            if text:
                units.append(_Unit(text, dict(doc.metadata), chapter, section, clause, not starts))
            buf.clear()

        for line in lines:
            if CHAPTER_RE.match(line):
                flush()
                chapter, section, clause, starts = _squash(line), "", "", True
            elif SECTION_RE.match(line):
                flush()
                section, clause, starts = _squash(line), "", True
            elif CLAUSE_RE.match(line):
                flush()
                clause, starts = _squash(line)[:CLAUSE_TITLE_CHARS], True
                buf.append(line)
            elif SUBCLAUSE_RE.match(line):
                flush()
                starts = True
                buf.append(line)
            else:
                buf.append(line)
        flush()
    return units


def split_by_structure(pages: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split loader output (one Document per page) into structure-aligned chunks of at most `chunk_size`."""
    units = _build_units(_clean_pages(pages))
    chunks: list[Document] = []

    def emit(body: str, first: _Unit, continued: bool) -> None:
        header = _header(first.chapter, first.section, first.clause, continued)
        metadata = dict(first.metadata)
        for key, value in (("chapter", first.chapter), ("section", first.section)):
            if value:
                metadata[key] = value
        chunks.append(Document(page_content=f"{header}\n{body}" if header else body, metadata=metadata))

    group: list[_Unit] = []
    used = 0

    def flush_group() -> None:
        nonlocal used
        if group:
            emit("\n".join(u.text for u in group), group[0], group[0].continued)
        group.clear()
        used = 0

    for unit in units:
        if group:
            first = group[0]
            same_place = (unit.metadata.get("page"), unit.chapter, unit.section) == \
                         (first.metadata.get("page"), first.chapter, first.section)
            budget = chunk_size - len(_header(first.chapter, first.section, first.clause, first.continued))
            if not same_place or used + 1 + len(unit.text) > budget:
                flush_group()

        budget = max(chunk_size - len(_header(unit.chapter, unit.section, unit.clause, True)), MIN_BODY_CHARS)
        if not group and len(unit.text) > budget:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=budget, chunk_overlap=min(chunk_overlap, budget // 4), separators=SEPARATORS,
            )
            for i, part in enumerate(splitter.split_text(unit.text)):
                emit(part, unit, unit.continued if i == 0 else True)
            continue

        group.append(unit)
        used += len(unit.text) + 1
    flush_group()
    return chunks
