"""IO utility functions for reading and writing data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_or_jsonl(path: str | Path) -> list[Any]:
    """
    Load data from a JSON array, compact JSONL, or indented JSONL file.

    Supported formats:
      - JSON array: ``[ {...}, {...}, ... ]``
      - Compact JSONL: one JSON object per line
      - Indented JSONL: pretty-printed JSON objects separated by blank lines

    Args:
        path: Path to the file

    Returns:
        List of data items

    Raises:
        ValueError: If file format is invalid
    """
    path = Path(path)
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return []

    # JSON array
    if txt.startswith("["):
        data = json.loads(txt)
        if not isinstance(data, list):
            raise ValueError("Input file must be a JSON array or JSONL with one item per line.")
        return data

    # Indented JSONL: records separated by blank lines
    if "\n\n" in txt:
        return [json.loads(block) for block in txt.split("\n\n") if block.strip()]

    # Compact JSONL: one JSON object per line
    return [json.loads(line) for line in txt.splitlines() if line.strip()]


def load_valid_records(path: str | Path) -> list[dict[str, Any]]:
    """
    Load valid JSON records from a file, tolerating a truncated trailing record.

    Useful for crash-recovery: if the last write was interrupted mid-record,
    all complete records before it are still returned.

    Args:
        path: Path to the file

    Returns:
        List of successfully parsed dict records
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return []

    blocks = txt.split("\n\n") if "\n\n" in txt else txt.splitlines()
    records: list[dict[str, Any]] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        try:
            obj = json.loads(block)
        except json.JSONDecodeError:
            # Truncated record from a crash — stop here
            break
        if isinstance(obj, dict):
            records.append(obj)
    return records


def chunked(lst: list[Any], size: int) -> list[list[Any]]:
    """
    Split a list into chunks of specified size.

    Args:
        lst: List to chunk
        size: Size of each chunk

    Returns:
        List of chunks
    """
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def write_jsonl(path: str | Path, records: list[Any], indent: int = 4) -> int:
    """
    Write records to JSONL file.

    Args:
        path: Output file path
        records: List of records to write
        indent: JSON indentation level

    Returns:
        Number of records written
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, indent=indent) + "\n\n")

    return len(records)
