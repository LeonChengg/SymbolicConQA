#!/usr/bin/env python3
"""CLI for checking Prolog entailment between context KB and SQ records.

Requires SWI-Prolog:
    sudo apt install swi-prolog

Usage:
    python scripts/check_entailment.py
    python scripts/check_entailment.py -c results/context_with_logic.jsonl \\
                                        -s results/SQ_with_logic.jsonl \\
                                        -o results/entailment_results.jsonl
"""

# NOTE: Do NOT add `from __future__ import annotations` here.
# Typer inspects annotations at runtime.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import typer

from symbolic_conqa.prolog_checker import run_entailment_check

app = typer.Typer(help="Check Prolog entailment for SQ records against context KB.")


@app.command()
def main(
    context_path: str = typer.Option(
        "results/context_with_logic.jsonl",
        "-c",
        "--context",
        help="Path to context_with_logic.jsonl",
    ),
    sq_path: str = typer.Option(
        "results/SQ_with_logic.jsonl",
        "-s",
        "--sq",
        help="Path to SQ_with_logic.jsonl",
    ),
    output_path: str = typer.Option(
        "results/entailment_results.jsonl",
        "-o",
        "--output",
        help="Output path for entailment results",
    ),
    timeout: int = typer.Option(
        10,
        "-t",
        "--timeout",
        help="Per-query SWI-Prolog timeout in seconds",
    ),
) -> None:
    """Check whether each SQ hypothesis is entailed by its context KB."""
    run_entailment_check(context_path, sq_path, output_path, timeout=timeout)


if __name__ == "__main__":
    app()
