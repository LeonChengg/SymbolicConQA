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


@app.command()
def main(
    task: Task = typer.Argument(..., help="Extraction task to run"),
    input_path: str | None = typer.Option(None, "-i", "--input", help="Input file path"),
    output_path: str | None = typer.Option(None, "-o", "--output", help="Output file path"),
    model: str = typer.Option("gpt-5-mini", "-m", "--model", help="OpenAI model"),
    batch_size: int = typer.Option(5, "-b", "--batch-size", help="Batch size"),
    num_batches: int | None = typer.Option(None, "-n", "--num-batches", help="Number of batches"),
    yes_no_only: bool = typer.Option(False, "--yes-no-only", help="Only process yes/no questions (scenario_question only)"),
) -> None:
    """Run logic extraction for a given task."""
    load_dotenv(find_dotenv())

    cfg = _TASK_CONFIGS[task]
    sample_filter = _is_yes_no if (yes_no_only and task == Task.scenario_question) else None
    run_extraction(
        text_extractor=_EXTRACTORS[task],
        in_path=input_path or str(cfg["default_input"]),
        out_path=output_path or str(cfg["default_output"]),
        model=model,
        batch_size=batch_size,
        num_test_batches=num_batches,
        include_hypothesis=bool(cfg["include_hypothesis"]),
        sample_filter=sample_filter,
    )


if __name__ == "__main__":
    app()
