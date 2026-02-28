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
    extract_text_from_semantic_tree,
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


class InputField(str, Enum):
    contents = "contents"
    semantic_tree = "semantic_tree"


_TASK_CONFIGS: dict[Task, dict[str, str | bool]] = {
    Task.context: {
        "default_input_contents": "data/ConditionalQA/v1_0/documents.json",
        "default_input_semantic_tree": "data/ConditionalQA/v1_0/documents_with_semantic_tree.json",
        "default_output": "results/context_with_logic.jsonl",
        "include_hypothesis": False,
    },
    Task.scenario_question: {
        "default_input_contents": "data/ConditionalQA/v1_0/dev.json",
        "default_input_semantic_tree": "data/ConditionalQA/v1_0/dev.json",
        "default_output": "results/SQ_with_logic.jsonl",
        "include_hypothesis": True,
    },
}

_EXTRACTORS: dict[tuple[Task, InputField], Any] = {
    (Task.context, InputField.contents): extract_text_from_contents,
    (Task.context, InputField.semantic_tree): extract_text_from_semantic_tree,
    (Task.scenario_question, InputField.contents): extract_text_from_scenario_question,
    (Task.scenario_question, InputField.semantic_tree): extract_text_from_scenario_question,
}


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
    input_field: InputField = typer.Option(
        InputField.contents,
        "--input-field",
        help=(
            "Which document field to use as input text for the LLM.\n\n"
            "'contents' (default): raw HTML strings from documents.json.\n\n"
            "'semantic_tree': structured semantic tree from "
            "documents_with_semantic_tree.json, rendered to clean plain text. "
            "Only applies to the 'context' task; 'scenario_question' always "
            "uses scenario+question text."
        ),
    ),
) -> None:
    """Run logic extraction for a given task."""
    load_dotenv(find_dotenv())

    cfg = _TASK_CONFIGS[task]
    sample_filter = _is_yes_no if (yes_no_only and task == Task.scenario_question) else None

    # Choose extractor and default input path based on task + input_field
    extractor = _EXTRACTORS[(task, input_field)]
    default_input_key = f"default_input_{input_field.value}"
    default_input = str(cfg[default_input_key])

    if input_field == InputField.semantic_tree and task == Task.context:
        typer.echo(
            f"Using semantic_tree renderer — default input: {default_input}"
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
        text_extractor=extractor,
        in_path=input_path or default_input,
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
