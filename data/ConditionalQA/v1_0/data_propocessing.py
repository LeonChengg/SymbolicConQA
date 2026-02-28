from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from bs4 import BeautifulSoup, Tag


# -----------------------------
# Heuristics (tuneable)
# -----------------------------
HEADING_LEVEL = {"h1": 1, "h2": 2, "h3": 3, "h4": 4}

ANY_INTRO_RE = re.compile(
    r"\b(either of the following apply|any of the following apply|one of the following)\b",
    re.I,
)
ALL_INTRO_RE = re.compile(
    r"\b(all of the following apply|both of the following apply)\b",
    re.I,
)

# A "rule-like" intro (very common in benefit policy pages)
ELIGIBLE_RE = re.compile(r"\b(you can (only )?make a new claim|you can claim|you can apply)\b", re.I)
NEGATIVE_RE = re.compile(r"\b(usually,\s*)?you will not|get .* if|cannot\b", re.I)

UNLESS_RE = re.compile(r"\bunless\b", re.I)
EXCEPT_RE = re.compile(r"\bexcept\b", re.I)


def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -----------------------------
# Table pre-processing
# Docs store tables as bare <tr>col1 | col2</tr> lines without a <table>
# wrapper and without <td>/<th> tags.  BeautifulSoup cannot parse these as
# proper table rows, so we normalise them first.
# -----------------------------
_TR_LINE_RE = re.compile(r"^\s*<tr[\s>]", re.IGNORECASE)


def _is_tr_line(line: str) -> bool:
    return bool(_TR_LINE_RE.match(line))


def _build_table_html(tr_lines: List[str]) -> str:
    """Convert a group of bare ``<tr>cell1 | cell2</tr>`` lines to a proper
    ``<table>`` element with ``<th>`` / ``<td>`` cells.

    The first row is promoted to a header row (``<th>``).
    Cells are delimited by ``|``.
    """
    rows_html: List[str] = []
    for idx, line in enumerate(tr_lines):
        m = re.search(r"<tr[^>]*>(.*?)</tr>", line, re.IGNORECASE | re.DOTALL)
        content = m.group(1).strip() if m else line.strip()
        cells = [c.strip() for c in content.split("|")]
        if idx == 0:
            # First row treated as header
            cells_html = "".join(f"<th>{c}</th>" for c in cells)
        else:
            cells_html = "".join(f"<td>{c}</td>" for c in cells)
        rows_html.append(f"<tr>{cells_html}</tr>")
    return "<table>" + "".join(rows_html) + "</table>"


def normalize_tr_rows(lines: List[str]) -> List[str]:
    """Replace runs of bare ``<tr>`` lines with a proper ``<table>`` block.

    Consecutive ``<tr>`` lines are grouped into a single table.  Non-``<tr>``
    lines are passed through unchanged.
    """
    result: List[str] = []
    i = 0
    while i < len(lines):
        if _is_tr_line(lines[i]):
            tr_group: List[str] = []
            while i < len(lines) and _is_tr_line(lines[i]):
                tr_group.append(lines[i])
                i += 1
            result.append(_build_table_html(tr_group))
        else:
            result.append(lines[i])
            i += 1
    return result


# -----------------------------
# Outline tree builder
# -----------------------------
def html_lines_to_soup(html_lines: List[str]) -> BeautifulSoup:
    # Normalise bare <tr> rows into proper <table> blocks before parsing.
    normalised = normalize_tr_rows(html_lines)
    # Wrap in a container to ensure valid parsing even if snippets have no root.
    html = "<div>\n" + "\n".join(normalised) + "\n</div>"
    return BeautifulSoup(html, "lxml")


def build_outline_tree(soup: BeautifulSoup) -> Dict[str, Any]:
    """
    Build an outline tree based on headings:
      - h1/h2/h3 create nested section nodes
      - p become text nodes
      - consecutive li nodes become list nodes
    """
    root = {"type": "root", "title": None, "children": []}
    stack: List[Tuple[int, Dict[str, Any]]] = [(0, root)]

    def current() -> Dict[str, Any]:
        return stack[-1][1]

    # We collect these tags in reading order
    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"]):
        if not isinstance(el, Tag):
            continue
        tag = el.name.lower()

        if tag in HEADING_LEVEL:
            title = clean_text(el.get_text(" ", strip=True))
            if not title:
                continue
            level = HEADING_LEVEL[tag]
            node = {"type": "section", "title": title, "level": level, "children": []}

            # pop until parent has smaller level
            while stack and stack[-1][0] >= level:
                stack.pop()
            current()["children"].append(node)
            stack.append((level, node))

        elif tag == "p":
            text = clean_text(el.get_text(" ", strip=True))
            if text:
                current()["children"].append({"type": "text", "text": text})

        elif tag == "li":
            text = clean_text(el.get_text(" ", strip=True))
            if not text:
                continue
            ch = current()["children"]
            if ch and ch[-1]["type"] == "list":
                ch[-1]["items"].append({"type": "item", "text": text})
            else:
                ch.append({"type": "list", "intro": None, "items": [{"type": "item", "text": text}]})

        elif tag == "table":
            # Optional: convert tables to row-objects, then attach
            table_node = table_to_node(el)
            if table_node is not None:
                current()["children"].append(table_node)

    return root


# -----------------------------
# Table normalization (optional but helpful)
# -----------------------------
def table_to_node(table: Tag) -> Optional[Dict[str, Any]]:
    """
    Convert a basic HTML table into a normalized structure.
    NOTE: This is a simplified parser (no rowspan/colspan support).
    If your tables have spans, I can provide an expanded version.
    """
    rows = table.find_all("tr")
    if not rows:
        return None

    grid: List[List[str]] = []
    for r in rows:
        cells = r.find_all(["th", "td"])
        grid.append([clean_text(c.get_text(" ", strip=True)) for c in cells])

    # Decide header: first row if it has <th> or looks like header-ish
    first_has_th = bool(rows[0].find_all("th"))
    if first_has_th:
        columns = grid[0]
        data_rows = grid[1:]
    else:
        # fall back to generic column names
        max_len = max(len(r) for r in grid)
        columns = [f"col_{i}" for i in range(max_len)]
        data_rows = [r + [""] * (max_len - len(r)) for r in grid]

    row_objs = []
    for r in data_rows:
        r2 = r + [""] * (len(columns) - len(r))
        row_objs.append({columns[i]: r2[i] for i in range(len(columns))})

    return {"type": "table", "columns": columns, "rows": row_objs}


# -----------------------------
# Semantic tree transformation
# -----------------------------
def infer_list_operator(intro_text: str) -> Optional[str]:
    if ANY_INTRO_RE.search(intro_text):
        return "ANY"
    if ALL_INTRO_RE.search(intro_text):
        return "ALL"
    return None


def is_rule_intro(text: str) -> bool:
    # Detect an intro sentence that likely governs subsequent conditions
    return bool(ELIGIBLE_RE.search(text) or NEGATIVE_RE.search(text))


def semanticize_outline_tree(outline: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert outline nodes into a semantic tree:
    - Attach intro paragraph to the subsequent list (as list.intro)
    - Add list.operator if intro suggests ANY/ALL
    - Wrap intro+list into a 'rule' node when intro looks like eligibility/disqualifier
    - Detect simple exceptions (unless/except) inside items
    """
    def process_node(node: Dict[str, Any]) -> Dict[str, Any]:
        if node["type"] in ("root", "section"):
            children = node.get("children", [])
            new_children: List[Dict[str, Any]] = []
            i = 0
            while i < len(children):
                ch = children[i]

                # Recurse sections
                if ch["type"] == "section":
                    new_children.append(process_node(ch))
                    i += 1
                    continue

                # Pattern: text intro + list => bind
                if ch["type"] == "text" and i + 1 < len(children) and children[i + 1]["type"] == "list":
                    intro = ch["text"]
                    lst = children[i + 1]

                    # Attach intro
                    lst2 = dict(lst)
                    lst2["intro"] = intro
                    op = infer_list_operator(intro)
                    if op:
                        lst2["operator"] = op

                    # Upgrade list items to conditions, split out exceptions
                    new_items: List[Dict[str, Any]] = []
                    exceptions: List[Dict[str, Any]] = []

                    for it in lst2["items"]:
                        t = it["text"]
                        if UNLESS_RE.search(t) or EXCEPT_RE.search(t):
                            exceptions.append({"type": "exception", "text": t})
                        else:
                            new_items.append({"type": "condition", "text": t})
                    lst2["items"] = new_items
                    if exceptions:
                        lst2["exceptions"] = exceptions

                    # Wrap into rule if intro is rule-like
                    if is_rule_intro(intro):
                        outcome = "ELIGIBLE" if not NEGATIVE_RE.search(intro) else "NOT_ELIGIBLE"
                        rule = {
                            "type": "rule",
                            "outcome": outcome,
                            "intro": intro,
                            "children": [lst2],
                        }
                        new_children.append(rule)
                    else:
                        # keep as text + list but already semanticized
                        new_children.append({"type": "text", "text": intro})
                        new_children.append(lst2)

                    i += 2
                    continue

                # Standalone list without explicit intro: keep
                if ch["type"] == "list":
                    lst2 = dict(ch)
                    # upgrade items
                    lst2["items"] = [{"type": "condition", "text": it["text"]} for it in lst2["items"]]
                    new_children.append(lst2)
                    i += 1
                    continue

                # Plain text, table, etc.
                new_children.append(ch)
                i += 1

            new_node = dict(node)
            new_node["children"] = new_children
            return new_node

        # Other nodes unchanged
        return node

    return process_node(outline)


# -----------------------------
# Utility: pretty print / save
# -----------------------------
def dump_json(obj: Any, path: Optional[str] = None) -> str:
    s = json.dumps(obj, indent=2, ensure_ascii=False)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(s)
    return s


# -----------------------------
# Document-level pipeline
# -----------------------------

def build_semantic_tree(contents: List[str]) -> Dict[str, Any]:
    """Convert a document’s ``contents`` list into a semantic tree.

    Steps:
    1. Normalise bare ``<tr>`` rows into proper ``<table>`` HTML.
    2. Parse with BeautifulSoup via :func:`html_lines_to_soup`.
    3. Build an outline tree (sections / text / list / table nodes).
    4. Enrich the outline into a semantic tree (rules / conditions / operators).

    Args:
        contents: List of HTML strings as found in ``documents.json``.

    Returns:
        A nested dict representing the semantic tree of the document.
    """
    soup = html_lines_to_soup(contents)
    outline = build_outline_tree(soup)
    return semanticize_outline_tree(outline)


def process_documents(
    input_path: str,
    output_path: str,
    key: str = "semantic_tree",
) -> None:
    """Read *input_path* (``documents.json``), enrich every record with a
    ``semantic_tree`` key, and write the result to *output_path*.

    The original record fields (``title``, ``url``, ``contents``) are kept
    intact; ``key`` is added (or overwritten) with the semantic tree produced
    by :func:`build_semantic_tree`.

    Args:
        input_path:  Path to the source ``documents.json``.
        output_path: Destination path for the enriched JSON file.
        key:         Name of the new field added to each record
                     (default ``"semantic_tree"``).
    """
    with open(input_path, encoding="utf-8") as f:
        docs: List[Dict[str, Any]] = json.load(f)

    enriched: List[Dict[str, Any]] = []
    for i, doc in enumerate(docs):
        contents = doc.get("contents", [])
        tree = build_semantic_tree(contents)
        enriched.append({**doc, key: tree})
        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1} / {len(docs)} documents …")

    dump_json(enriched, output_path)
    print(f"Written {len(enriched)} enriched records to {output_path}")


# -----------------------------
# CLI entry point
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build semantic trees from documents.json HTML content and write "
            "an enriched JSON file with a ‘semantic_tree’ key per document."
        )
    )
    parser.add_argument(
        "-i", "--input",
        default="data/ConditionalQA/v1_0/documents.json",
        help="Path to documents.json (default: data/ConditionalQA/v1_0/documents.json)",
    )
    parser.add_argument(
        "-o", "--output",
        default="data/ConditionalQA/v1_0/documents_with_semantic_tree.json",
        help=(
            "Output path for the enriched JSON file "
            "(default: data/ConditionalQA/v1_0/documents_with_semantic_tree.json)"
        ),
    )
    parser.add_argument(
        "--key",
        default="semantic_tree",
        help="Name of the new field added to each record (default: semantic_tree)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the built-in example and print the resulting trees instead of processing documents.json",
    )
    args = parser.parse_args()

    if args.demo:
        # Built-in smoke-test example
        html_lines = [
            "<h1>Eligibility</h1>",
            "<p>Housing Benefit can help you pay your rent if you’re unemployed, on a low income or claiming benefits. It’s being replaced by Universal Credit.</p>",
            "<p>You can only make a new claim for Housing Benefit if either of the following apply:</p>",
            "<li>you have reached State Pension age</li>",
            "<li>you’re in supported, sheltered or temporary housing</li>",
            "<h2>You’ve reached State Pension age</h2>",
            "<p>If you’re single you can make a new claim for Housing Benefit.</p>",
            # Bare-<tr> table (as found in documents.json)
            "<tr>Rate | Amount</tr>",
            "<tr>Standard | £100 per week</tr>",
            "<tr>Enhanced | £150 per week</tr>",
        ]
        soup = html_lines_to_soup(html_lines)
        outline = build_outline_tree(soup)
        semantic = semanticize_outline_tree(outline)
        print("=== OUTLINE ===")
        print(dump_json(outline))
        print("\n=== SEMANTIC ===")
        print(dump_json(semantic))
    else:
        process_documents(args.input, args.output, key=args.key)