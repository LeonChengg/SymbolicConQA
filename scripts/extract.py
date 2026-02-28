#!/usr/bin/env python3
"""Unified CLI for logic extraction."""

# NOTE: Do NOT use `from __future__ import annotations` here.
# Typer inspects annotations at runtime; PEP 563 deferred evaluation breaks it.

import sys
from enum import Enum
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import typer

from typing import Any

from symbolic_conqa.extraction import (
    extract_text_from_contents,
    extract_text_from_scenario_question,
    run_extraction,
)


def _is_yes_no(sample: dict[str, Any]) -> bool:
    """Return True if the sample's first answer is 'yes' or 'no'."""
    answers = sample.get("answers", [])
    if not answers:
        return False
    first_answer = answers[0][0] if answers[0] else ""
    return first_answer.lower() in ("yes", "no")

app = typer.Typer(help="Extract logic from text using LLMs.")


def _build_context_predicate_index(context_kb_path: str) -> dict[str, str]:
    """Load a context KB file and build {url: formatted_predicates_str}."""
    from symbolic_conqa.io_utils import load_json_or_jsonl

    records = load_json_or_jsonl(context_kb_path)
    index: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        # The URL lives in the nested data dict
        url = record.get("data", {}).get("url", "") or record.get("url", "")
        if not url:
            continue
        predicates = record.get("logic_kb", {}).get("predicates", [])
        if not predicates:
            continue
        formatted = "\n".join(
            f"- {p['name']}/{p['arity']} — {p.get('gloss', '')}"
            for p in predicates
        )
        index[url] = formatted
    return index


def _build_context_constant_index(context_kb_path: str) -> dict[str, str]:
    """Load a context KB file and build {url: formatted_constants_str}.

    Includes all constants for standardization.
    """
    from symbolic_conqa.io_utils import load_json_or_jsonl

    records = load_json_or_jsonl(context_kb_path)
    index: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        url = record.get("data", {}).get("url", "") or record.get("url", "")
        if not url:
            continue
        constants = record.get("logic_kb", {}).get("constants", [])
        if not constants:
            continue
        formatted = "\n".join(
            f"- {c['id']} — {c.get('gloss', '')} (type: {c.get('type', 'unknown')})"
            for c in constants
        )
        index[url] = formatted
    return index


def _build_context_rules_index(context_kb_path: str) -> dict[str, str]:
    """Load a context KB file and build {url: formatted Prolog rules}."""
    from symbolic_conqa.io_utils import load_json_or_jsonl

    records = load_json_or_jsonl(context_kb_path)
    index: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        url = record.get("data", {}).get("url", "") or record.get("url", "")
        if not url:
            continue
        rules = record.get("logic_kb", {}).get("prolog", {}).get("rules", [])
        if not rules:
            continue
        index[url] = "\n".join(rules)
    return index


class Task(str, Enum):
    context = "context"
    scenario_question = "scenario_question"


_TASK_CONFIGS: dict[Task, dict[str, str | bool]] = {
    Task.context: {
        "default_input": "data/ConditionalQA/v1_0/documents.json",
        "default_output": "results/context_with_logic.jsonl",
        "include_hypothesis": False,
    },
    Task.scenario_question: {
        "default_input": "data/ConditionalQA/v1_0/dev.json",
        "default_output": "results/SQ_with_logic.jsonl",
        "include_hypothesis": True,
    },
}

_EXTRACTORS = {
    Task.context: extract_text_from_contents,
    Task.scenario_question: extract_text_from_scenario_question,
}


def _collect_urls_from_file(path: str) -> set[str]:
    """Load a JSON/JSONL file and collect all unique ``url`` fields."""
    from symbolic_conqa.io_utils import load_json_or_jsonl

    records = load_json_or_jsonl(path)
    urls: set[str] = set()
    for rec in records:
        if isinstance(rec, dict):
            url = rec.get("url", "")
            if url:
                urls.add(url)
    return urls


@app.command()
def main(
    task: Task = typer.Argument(..., help="Extraction task to run"),
    input_path: str | None = typer.Option(None, "-i", "--input", help="Input file path"),
    output_path: str | None = typer.Option(None, "-o", "--output", help="Output file path"),
    model: str = typer.Option("gpt-5-mini", "-m", "--model", help="OpenAI model"),
    batch_size: int = typer.Option(5, "-b", "--batch-size", help="Batch size"),
    num_batches: int | None = typer.Option(None, "-n", "--num-batches", help="Number of batches"),
    yes_no_only: bool = typer.Option(False, "--yes-no-only", help="Only process yes/no questions (scenario_question only)"),
    context_kb: str | None = typer.Option(None, "--context-kb", help="Path to context KB JSONL for injecting predicates (scenario_question only)"),
    filter_by: str | None = typer.Option(None, "--filter-by", help="Path to a JSON/JSONL file (e.g. dev.json) to filter context documents by URL (context task only)"),
) -> None:
    """Run logic extraction for a given task."""
    load_dotenv(find_dotenv())

    cfg = _TASK_CONFIGS[task]
    sample_filter = _is_yes_no if (yes_no_only and task == Task.scenario_question) else None

    # Build URL filter for context task
    if filter_by and task == Task.context:
        url_set = _collect_urls_from_file(filter_by)
        typer.echo(f"Filtering context documents to {len(url_set)} unique URLs from {filter_by}")
        base_filter = sample_filter
        sample_filter = lambda s, _urls=url_set, _base=base_filter: (
            s.get("url", "") in _urls and (_base is None or _base(s))
        )

    # Build context predicate, constant, and rules indices if provided
    ctx_pred_index: dict[str, str] | None = None
    ctx_const_index: dict[str, str] | None = None
    ctx_rules_index: dict[str, str] | None = None
    sample_key_fn = None
    if context_kb and task == Task.scenario_question:
        ctx_pred_index = _build_context_predicate_index(context_kb)
        ctx_const_index = _build_context_constant_index(context_kb)
        ctx_rules_index = _build_context_rules_index(context_kb)
        sample_key_fn = lambda s: s.get("url", "")
        typer.echo(f"Loaded context predicates for {len(ctx_pred_index)} URLs from {context_kb}")
        typer.echo(f"Loaded context constants for {len(ctx_const_index)} URLs from {context_kb}")
        typer.echo(f"Loaded context rules for {len(ctx_rules_index)} URLs from {context_kb}")

    run_extraction(
        text_extractor=_EXTRACTORS[task],
        in_path=input_path or str(cfg["default_input"]),
        out_path=output_path or str(cfg["default_output"]),
        model=model,
        batch_size=batch_size,
        num_test_batches=num_batches,
        include_hypothesis=bool(cfg["include_hypothesis"]),
        sample_filter=sample_filter,
        context_predicates_by_key=ctx_pred_index,
        context_constants_by_key=ctx_const_index,
        context_rules_by_key=ctx_rules_index,
        sample_key_fn=sample_key_fn,
    )


if __name__ == "__main__":
    app()
