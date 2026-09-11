"""Section-aware chunking for Markdown and plain text (Arabic + English).

Every chunk keeps its document title and section path so answers can cite
``document + section`` for every claim.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
# Plain-text heading heuristic: short line, optionally numbered/ornamented.
_TXT_HEADING_RE = re.compile(r"^(#{1,6}\s+)?(\d+[.\-)\s]+|[*\-#=\s>]*)(.{3,80})$")


@dataclass
class Chunk:
    doc: str      # document title (filename stem or first H1)
    path: str     # source file path as given at ingest
    section: str  # "Section > Subsection" breadcrumb, "" if none
    text: str
    lang: str     # "ar" if predominantly Arabic, else "en"


def detect_lang(text: str) -> str:
    arabic = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")
    letters = sum(1 for ch in text if ch.isalpha())
    if letters and arabic / letters > 0.25:
        return "ar"
    return "en"


@dataclass
class _Section:
    level: int
    title: str
    lines: list[str]


def _split_markdown(text: str) -> tuple[str, list[_Section]]:
    """Split markdown into sections; returns (doc_title, sections)."""
    doc_title = ""
    sections: list[_Section] = []
    current = _Section(level=0, title="", lines=[])
    for line in text.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            if current.lines or current.title:
                sections.append(current)
            level = len(m.group(1))
            title = m.group(2).strip()
            if level == 1 and not doc_title:
                doc_title = title
            current = _Section(level=level, title=title, lines=[])
        else:
            current.lines.append(line)
    if current.lines or current.title:
        sections.append(current)
    return doc_title, sections


def _section_path(stack: list[tuple[int, str]], level: int, title: str) -> str:
    """Breadcrumb of enclosing sections, excluding the H1 document title.

    The H1 is the document title (stored on ``Chunk.doc``), so breadcrumbs
    start at H2. An H1 section with body text but no subsections still yields
    "" (untitled top-level section).
    """
    while stack and stack[-1][0] >= level:
        stack.pop()
    if title and level > 1:
        stack.append((level, title))
    elif title and not stack:
        # H1-level content: track for nesting but don't show in breadcrumb.
        stack.append((level, title))
    visible = [t for lvl, t in stack if lvl > 1]
    return " > ".join(visible)


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    """Hard-split an oversized paragraph on word boundaries with overlap."""
    parts: list[str] = []
    rest = text.strip()
    while len(rest) > max_chars:
        cut = rest.rfind(" ", 0, max_chars)
        cut = cut if cut > max_chars // 2 else max_chars
        parts.append(rest[:cut].strip())
        rest = rest[max(0, cut - overlap):].strip()
    if rest:
        parts.append(rest)
    return parts


def _pack_paragraphs(paragraphs: list[str], max_chars: int, overlap: int) -> list[str]:
    """Greedily pack paragraphs into chunks of <= max_chars with char overlap."""
    units: list[str] = []
    for para in paragraphs:
        if para:
            units.extend(_split_long(para, max_chars, overlap))
    chunks: list[str] = []
    buf = ""
    for unit in units:
        candidate = (buf + "\n\n" + unit).strip() if buf else unit
        if len(candidate) <= max_chars:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap > 0 else ""
            joined = (tail + "\n\n" + unit).strip() if tail else unit
            buf = joined if len(joined) <= max_chars else unit
        else:
            chunks.append(unit)  # unit already <= max_chars by construction
            buf = ""
    if buf.strip():
        chunks.append(buf.strip())
    return [c for c in chunks if c.strip()]


def chunk_markdown(text: str, doc: str, path: str,
                   max_chars: int = 1200, overlap: int = 150) -> list[Chunk]:
    doc_title, sections = _split_markdown(text)
    title = doc_title or doc
    stack: list[tuple[int, str]] = []
    out: list[Chunk] = []
    for sec in sections:
        breadcrumb = _section_path(stack, sec.level or 2, sec.title)
        body = "\n".join(sec.lines).strip()
        if not body:
            continue
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        for piece in _pack_paragraphs(paragraphs, max_chars, overlap):
            out.append(Chunk(doc=title, path=path, section=breadcrumb,
                             text=piece, lang=detect_lang(piece)))
    return out


def chunk_text(text: str, doc: str, path: str,
               max_chars: int = 1200, overlap: int = 150) -> list[Chunk]:
    """Plain-text chunking with a heading heuristic (works for Arabic too)."""
    sections: list[tuple[str, list[str]]] = []
    cur_title, cur_lines = "", []
    for line in text.splitlines():
        s = line.strip()
        m = _TXT_HEADING_RE.match(s) if s and len(s) <= 80 else None
        head = m.group(3).strip() if m else s
        is_heading = bool(m and not s.endswith((".", ":", "،", ".", "؟", "?"))
                          and len(head.split()) <= 10)
        if is_heading and cur_lines:
            sections.append((cur_title, cur_lines))
            cur_title, cur_lines = head, []
        elif is_heading:
            cur_title = head
        else:
            cur_lines.append(line)
    sections.append((cur_title, cur_lines))
    out: list[Chunk] = []
    for title, lines in sections:
        body = "\n".join(lines).strip()
        if not body:
            continue
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        for piece in _pack_paragraphs(paragraphs, max_chars, overlap):
            out.append(Chunk(doc=doc, path=path, section=title,
                             text=piece, lang=detect_lang(piece)))
    return out


def load_file(path: str) -> tuple[str, str]:
    """Return (kind, text) for a supported file. PDF needs pypdf (optional)."""
    lower = path.lower()
    if lower.endswith(".md"):
        with open(path, encoding="utf-8") as fh:
            return "md", fh.read()
    if lower.endswith(".txt"):
        with open(path, encoding="utf-8") as fh:
            return "txt", fh.read()
    if lower.endswith(".pdf"):
        try:
            from pypdf import PdfReader  # optional dependency
        except ImportError as exc:
            raise RuntimeError(
                f"PDF support needs the optional 'pypdf' package: {exc}") from exc
        reader = PdfReader(path)
        return "txt", "\n\n".join((page.extract_text() or "") for page in reader.pages)
    raise ValueError(f"unsupported file type: {path} (use .md or .txt)")


def chunk_file(path: str, max_chars: int = 1200, overlap: int = 150) -> list[Chunk]:
    kind, text = load_file(path)
    doc = os.path.splitext(os.path.basename(path))[0]
    if kind == "md":
        chunks = chunk_markdown(text, doc, path, max_chars, overlap)
    else:
        chunks = chunk_text(text, doc, path, max_chars, overlap)
    # Fallback: never return zero chunks for a non-empty file.
    if not chunks and text.strip():
        chunks = [Chunk(doc=doc, path=path, section="",
                        text=text.strip()[:max_chars], lang=detect_lang(text))]
    return chunks
