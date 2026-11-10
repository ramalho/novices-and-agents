#!/usr/bin/env python3
"""Convert the ACM-style PDF in this directory to semantic Markdown.

The converter deliberately favours simple, meaningful markup over reproducing
the visual layout: headings become ATX headings, paragraphs become paragraphs,
footnote markers become Markdown footnotes, and the reference list stays a
list of labelled entries.

Requires PyMuPDF:

    python3 -m pip install pymupdf

Usage:

    python3 pdf_to_markdown.py novice-programming-agentic.pdf
    python3 pdf_to_markdown.py paper.pdf -o paper.md
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

try:  # PyMuPDF 1.24+ exposes the modern name; older wheels only expose fitz.
    import pymupdf
except ImportError:  # pragma: no cover - compatibility with old PyMuPDF
    import fitz as pymupdf  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


@dataclass
class Span:
    text: str
    size: float
    flags: int


@dataclass
class Line:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float
    spans: list[Span] = field(default_factory=list)

    @property
    def bold(self) -> bool:
        visible = [s for s in self.spans if s.text.strip()]
        return bool(visible) and all(s.flags & 16 for s in visible)

    @property
    def italic(self) -> bool:
        visible = [s for s in self.spans if s.text.strip()]
        return bool(visible) and all(s.flags & 2 for s in visible)


def extract_lines(page: "pymupdf.Page") -> list[Line]:
    """Return the non-empty text lines on a page, with font information."""
    lines: list[Line] = []
    data = page.get_text("dict")
    for block in data["blocks"]:
        if block["type"] != 0:  # image
            continue
        for raw_line in block["lines"]:
            spans = [
                Span(
                    text=s["text"],
                    size=float(s["size"]),
                    flags=int(s["flags"]),
                )
                for s in raw_line["spans"]
                if s["text"]
            ]
            text = "".join(s.text for s in spans)
            if not text.strip():
                continue
            x0, y0, x1, y1 = (float(v) for v in raw_line["bbox"])
            lines.append(
                Line(
                    text=text,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    size=max((s.size for s in spans), default=0.0),
                    spans=spans,
                )
            )
    return lines


# ---------------------------------------------------------------------------
# Page furniture
# ---------------------------------------------------------------------------


FOOTNOTE_DEF_RE = re.compile(r"^(\d+)\s*(https?://\S+)$")
AUTHOR_RE = re.compile(r"^([A-Z][A-Z .'\-]{2,}),\s*(.+)$")


def is_page_furniture(line: Line, page_height: float, page_number: int) -> bool:
    text = line.text.strip()

    # Running heads and folios live in the top and bottom margins.  Page one
    # has no running head (its title sits just below the normal head line).
    if page_number > 1 and line.y0 < page_height * 0.11:
        return True
    if line.y0 > page_height * 0.83:
        return True
    if re.fullmatch(r"Manuscript submitted to ACM(?:\s+\d+)?", text):
        return True
    if text.startswith("This work is licensed under"):
        return True
    if re.match(r"^©\s*\d{4}\s+Copyright held", text):
        return True
    if text.startswith("Authors’ Contact Information:"):
        return True
    if text.startswith("USA;") and line.size < 8:
        return True
    return False


def merge_same_baseline(lines: list[Line]) -> list[Line]:
    """Join pieces that PyMuPDF reports separately but share a baseline.

    The main example is the section number (``1``) and section title
    (``Introduction``), which are separate text lines in the PDF.
    """
    if not lines:
        return []

    ordered = sorted(lines, key=lambda line: (round(line.y0, 1), line.x0))
    merged: list[Line] = []
    for line in ordered:
        if (
            merged
            and abs(line.y0 - merged[-1].y0) < 1.5
            and abs(line.size - merged[-1].size) < 0.6
            and line.bold == merged[-1].bold
        ):
            previous = merged[-1]
            gap = line.x0 - previous.x1
            joiner = " " if gap > 0.5 else ""
            previous.text = previous.text.rstrip() + joiner + line.text.lstrip()
            previous.x1 = max(previous.x1, line.x1)
            previous.y1 = max(previous.y1, line.y1)
            if joiner:
                previous.spans.append(Span(" ", line.size, 0))
            previous.spans.extend(line.spans)
        else:
            merged.append(line)
    return merged


# ---------------------------------------------------------------------------
# Paragraphs and headings
# ---------------------------------------------------------------------------


@dataclass
class Block:
    kind: str  # "heading", "paragraph", or "reference"
    lines: list[Line]

    @property
    def text(self) -> str:
        return " ".join(line.text.strip() for line in self.lines)


def is_section_heading(line: Line) -> bool:
    text = line.text.strip()
    if not line.bold or line.size < 8.8:
        return False
    return bool(re.match(r"^\d+\s+\S", text)) or text == "References"


def build_blocks(
    lines: list[Line], in_references: bool = False
) -> tuple[list[Block], bool]:
    """Group lines into paragraphs, using first-line indentation as the cue."""
    if not lines:
        return [], in_references

    body_lines = lines if in_references else [line for line in lines if line.size >= 8.5]
    if not body_lines:
        return [], in_references
    margin = min(line.x0 for line in body_lines)

    blocks: list[Block] = []
    current: list[Line] = []

    def flush() -> None:
        nonlocal current
        if current:
            kind = "reference" if in_references else "paragraph"
            blocks.append(Block(kind=kind, lines=current))
            current = []

    for line in lines:
        if is_section_heading(line):
            flush()
            blocks.append(Block(kind="heading", lines=[line]))
            in_references = line.text.strip() == "References"
            continue

        if in_references:
            if re.match(r"^\[\d+\]", line.text.strip()):
                flush()
                current = [line]
            else:
                current.append(line)
            continue

        if current and line.x0 > margin + 4.0:
            flush()
            current = [line]
        else:
            current.append(line)

    flush()
    return blocks, in_references


def merge_cross_page_paragraphs(blocks: list[Block]) -> list[Block]:
    """Reattach a paragraph that was interrupted by a page boundary."""
    merged: list[Block] = []
    for block in blocks:
        if (
            merged
            and block.kind == "paragraph"
            and merged[-1].kind == "paragraph"
            and not merged[-1].text.rstrip().endswith((".", "!", "?", ":", ";"))
            and block.text[:1].islower()
        ):
            merged[-1].lines.extend(block.lines)
        else:
            merged.append(block)
    return merged


# ---------------------------------------------------------------------------
# Inline Markdown
# ---------------------------------------------------------------------------

# Prefixes that normally keep their hyphen when a line is wrapped.
COMPOUND_PREFIXES = {
    "ai",
    "anti",
    "auto",
    "bio",
    "co",
    "code",
    "cross",
    "end",
    "hyper",
    "inter",
    "intra",
    "macro",
    "meta",
    "micro",
    "multi",
    "non",
    "out",
    "over",
    "post",
    "pre",
    "re",
    "self",
    "semi",
    "sub",
    "super",
    "trans",
    "ultra",
    "under",
    "well",
}

_WORDS: set[str] | None = None


def is_dictionary_word(word: str) -> bool:
    global _WORDS
    if _WORDS is None:
        _WORDS = set()
        for path in (Path("/usr/share/dict/words"),):
            try:
                _WORDS = {w.strip().lower() for w in path.read_text().splitlines()}
                break
            except OSError:
                continue
    return word.lower() in _WORDS


def keeps_hyphen(fragment: str) -> bool:
    fragment = fragment.strip(".,;:!?()[]{}\"'“”‘’").lower()
    if not fragment:
        return False
    if fragment in COMPOUND_PREFIXES:
        return True
    if is_dictionary_word(fragment):
        return True
    # Fallback for machines without /usr/share/dict/words: long fragments are
    # much more likely to be complete words than syllable splits.
    return len(fragment) >= 8


def wrap_emphasis(text: str, marker: str) -> str:
    """Wrap the non-space core of a span, leaving its spaces outside."""
    core = text.strip()
    if not core:
        return text
    leading = text[: len(text) - len(text.lstrip())]
    trailing = text[len(text.rstrip()) :]
    return f"{leading}{marker}{core}{marker}{trailing}"


def line_to_markdown(line: Line) -> str:
    runs: list[tuple[str, str]] = []

    def add(style: str, text: str) -> None:
        if not text:
            return
        if runs and runs[-1][0] == style:
            runs[-1] = (style, runs[-1][1] + text)
        else:
            runs.append((style, text))

    for span in line.spans:
        text = span.text
        if not text:
            continue
        if span.flags & 1 and text.strip().isdigit():
            add("footnote", text.strip())
        elif span.flags & 16:
            add("bold", text)
        elif span.flags & 2:
            add("italic", text)
        else:
            add("plain", text)

    pieces: list[str] = []
    previous_style: str | None = None
    previous_text = ""
    for style, text in runs:
        # PyMuPDF occasionally drops the space after a maths glyph (e.g.
        # "𝑁times").  If a styled run is followed directly by a word, restore
        # the space that the visual gap represents.
        if (
            previous_style in {"bold", "italic"}
            and style == "plain"
            and text[:1].isalnum()
            and not previous_text.endswith((" ", "\t"))
        ):
            pieces.append(" ")

        if style == "footnote":
            pieces.append(f"[^{text}]")
        elif style == "bold":
            pieces.append(wrap_emphasis(text, "**"))
        elif style == "italic":
            pieces.append(wrap_emphasis(text, "*"))
        else:
            pieces.append(text)

        previous_style = style
        previous_text = text

    result = "".join(pieces)
    result = result.replace("\u00a0", " ")
    result = re.sub(r"[ \t]+", " ", result).strip()
    return result


def join_wrapped_lines(lines: list[str], reference: bool = False) -> str:
    if not lines:
        return ""
    result = lines[0]
    for next_line in lines[1:]:
        next_line = next_line.strip()
        if not next_line:
            continue
        if result.endswith("-") and next_line[:1].islower():
            fragment = result.rsplit(" ", 1)[-1].rstrip("-")
            if keeps_hyphen(fragment):
                result += next_line
            else:
                result = result[:-1] + next_line
        elif reference and re.search(r"(?:https?://|doi:)\S*\.$", result):
            result += next_line
        else:
            result += " " + next_line
    return result


def tidy_emphasis(text: str) -> str:
    """Merge emphasis runs that were split by a line or font boundary."""
    while True:
        updated = re.sub(
            r"\*\*([^*\n]+)\*\* \*\*([^*\n]+)\*\*",
            r"**\1 \2**",
            text,
        )
        updated = re.sub(
            r"\*([^*\n]+)\* \*([^*\n]+)\*",
            r"*\1 \2*",
            updated,
        )
        if updated == text:
            return text
        text = updated


def block_to_markdown(block: Block) -> str:
    if block.kind == "heading":
        return "## " + block.lines[0].text.strip()
    rendered = [line_to_markdown(line) for line in block.lines]
    return tidy_emphasis(
        join_wrapped_lines(rendered, reference=(block.kind == "reference"))
    )


# ---------------------------------------------------------------------------
# Front matter
# ---------------------------------------------------------------------------


def format_authors(authors: list[tuple[str, str]]) -> str:
    if not authors:
        return ""

    names = [
        f"**{name.title() if name.isupper() else name}**"
        for name, _ in authors
    ]
    if len(names) == 1:
        joined = names[0]
    elif len(names) == 2:
        joined = f"{names[0]} and {names[1]}"
    else:
        joined = ", ".join(names[:-1]) + ", and " + names[-1]

    affiliations = {affiliation for _, affiliation in authors}
    if len(affiliations) == 1:
        return joined + "\n\n" + affiliations.pop()
    return "\n\n".join(
        f"**{name.title() if name.isupper() else name}** — {affiliation}"
        for name, affiliation in authors
    )


def find_line(lines: list[Line], prefix: str, start: int = 0) -> int | None:
    for index in range(start, len(lines)):
        if lines[index].text.strip().startswith(prefix):
            return index
    return None


def parse_front_matter(
    lines: list[Line],
) -> tuple[str, list[tuple[str, str]], str, str, str, str, list[Line]]:
    """Split the title page into title, authors, abstract, and metadata."""
    title_index = next(
        (i for i, line in enumerate(lines) if line.size >= 11.0 and line.bold),
        0,
    )
    title = lines[title_index].text.strip()

    author_lines: list[tuple[str, str]] = []
    index = title_index + 1
    while index < len(lines):
        match = AUTHOR_RE.match(lines[index].text.strip())
        if not match:
            break
        author_lines.append((match.group(1).strip(), match.group(2).strip()))
        index += 1

    ccs_index = find_line(lines, "CCS Concepts:", index)
    keywords_index = find_line(lines, "Additional Key Words and Phrases:", index)
    acm_index = find_line(lines, "ACM Reference Format:", index)

    abstract_end = ccs_index or keywords_index or acm_index or len(lines)
    abstract = tidy_emphasis(
        join_wrapped_lines(
            [line_to_markdown(line) for line in lines[index:abstract_end]]
        )
    )

    ccs = ""
    if ccs_index is not None:
        ccs_end = keywords_index or acm_index or len(lines)
        ccs = tidy_emphasis(
            join_wrapped_lines(
                [line_to_markdown(line) for line in lines[ccs_index:ccs_end]]
            )
        )
        ccs = re.sub(r"^CCS Concepts:\s*", "**CCS Concepts:** ", ccs)
        ccs = re.sub(r"\s*→\s*", " → ", ccs)

    keywords = ""
    if keywords_index is not None:
        keywords_end = acm_index or len(lines)
        keywords = tidy_emphasis(
            join_wrapped_lines(
                [
                    line_to_markdown(line)
                    for line in lines[keywords_index:keywords_end]
                ]
            )
        )
        keywords = re.sub(
            r"^Additional Key Words and Phrases:\s*",
            "**Additional Key Words and Phrases:** ",
            keywords,
        )

    acm = ""
    body_start = len(lines)
    if acm_index is not None:
        body_start = next(
            (
                i
                for i in range(acm_index, len(lines))
                if is_section_heading(lines[i])
            ),
            len(lines),
        )
        acm = tidy_emphasis(
            join_wrapped_lines(
                [line_to_markdown(line) for line in lines[acm_index:body_start]]
            )
        )

    return title, author_lines, abstract, ccs, keywords, acm, lines[body_start:]


# ---------------------------------------------------------------------------
# Document conversion
# ---------------------------------------------------------------------------


def convert(pdf_path: Path) -> str:
    document = pymupdf.open(pdf_path)
    footnotes: dict[int, str] = {}
    page_lines: list[list[Line]] = []

    for page_number, page in enumerate(document, start=1):
        lines = extract_lines(page)
        kept: list[Line] = []
        for line in lines:
            if is_page_furniture(line, page.rect.height, page_number):
                continue
            footnote = FOOTNOTE_DEF_RE.match(line.text.strip())
            if footnote and line.size < 8.5:
                footnotes[int(footnote.group(1))] = footnote.group(2)
                continue
            kept.append(line)
        page_lines.append(merge_same_baseline(kept))

    if not page_lines:
        return ""

    title, authors, abstract, ccs, keywords, acm, first_body = parse_front_matter(
        page_lines[0]
    )

    blocks: list[Block] = []
    blocks.extend(build_blocks(first_body)[0])

    in_references = any(
        block.kind == "heading" and block.lines[0].text.strip() == "References"
        for block in blocks
    )
    for lines in page_lines[1:]:
        page_blocks, in_references = build_blocks(lines, in_references)
        blocks.extend(page_blocks)

    blocks = merge_cross_page_paragraphs(blocks)

    parts: list[str] = [f"# {title}"]
    author_block = format_authors(authors)
    if author_block:
        parts.append(author_block)
    if abstract:
        parts.append("## Abstract")
        parts.append(abstract)
    for metadata in (ccs, keywords, acm):
        if metadata:
            parts.append(metadata)

    for block in blocks:
        parts.append(block_to_markdown(block))

    if footnotes:
        parts.append(
            "\n".join(
                f"[^{number}]: {url}"
                for number, url in sorted(footnotes.items())
            )
        )

    return "\n\n".join(part for part in parts if part).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an academic PDF to simple semantic Markdown."
    )
    parser.add_argument("pdf", type=Path, help="input PDF file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output Markdown file (default: input name with .md)",
    )
    args = parser.parse_args()

    output = args.output or args.pdf.with_suffix(".md")
    markdown = convert(args.pdf)
    output.write_text(markdown, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
