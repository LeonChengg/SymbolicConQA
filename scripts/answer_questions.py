#!/usr/bin/env python3
"""Answer questions in B100_question_with_logic.jsonl using context KB from
all_context_with_logic.jsonl via SWI-Prolog entailment.

For each question record the script:
  1. Looks up the matching context KB by ``data.url``.
  2. Merges context facts/rules with the scenario facts/rules.
  3. Uses the pre-extracted ``logic_kb.prolog.hypothesis`` as the Prolog goal.
  4. Runs the goal against SWI-Prolog:
       - Ground goals  (no free variables): True → "yes", False → "no"
       - Variable goals (e.g. ``time_to_hear_back(me, T)``):
         all solutions for the free variable(s) are collected and returned.
  5. Writes results to an output JSONL file and prints a summary.

Requires SWI-Prolog:
    sudo apt install swi-prolog   # Debian/Ubuntu
    conda install -c conda-forge swi-prolog

Usage:
    python scripts/answer_questions.py
    python scripts/answer_questions.py \\
        -c results/all_context_with_logic.jsonl \\
        -s results/B100_question_with_logic.jsonl \\
        -o results/B100_answers.jsonl
"""

# NOTE: Do NOT add `from __future__ import annotations` here.
# Typer inspects annotations at runtime.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from symbolic_conqa.io_utils import load_json_or_jsonl, write_jsonl
from symbolic_conqa.prolog_checker import (
    build_prolog_program,
    extract_prolog_kb,
    load_context_index,
    query_prolog_bindings,
)

console = Console()
app = typer.Typer(help="Answer questions using Prolog entailment.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _question_type(sq_data: dict) -> str:
    """Classify a question as 'yesno', 'value', or 'not_answerable'."""
    if sq_data.get("not_answerable"):
        return "not_answerable"
    gold_values = [a[0].strip().lower() for a in sq_data.get("answers", [])]
    if gold_values and all(v in ("yes", "no") for v in gold_values):
        return "yesno"
    return "value"


def _predict(
    entails: bool | None,
    values: list[str],
    q_type: str,
) -> str | list[str] | None:
    """Map an entailment result to a predicted answer string."""
    if entails is None:
        return None                       # error / timeout
    if entails:
        return values if values else "yes"  # extracted bindings or ground True
    # entails is False
    return "not answerable" if q_type == "not_answerable" else "no"


# ---------------------------------------------------------------------------
# Evaluation summary
# ---------------------------------------------------------------------------


def _print_summary(results: list[dict]) -> None:
    total = len(results)
    no_ctx = sum(1 for r in results if not r["context_found"])
    entailed = sum(1 for r in results if r["entails"] is True)
    not_entailed = sum(1 for r in results if r["entails"] is False)
    errors = sum(1 for r in results if r["entails"] is None)

    console.print(f"\n[bold]── Overall ({total} questions) ──[/bold]")
    console.print(f"  Context match  : {total - no_ctx} / {total}")
    console.print(f"  Entailed       : {entailed}")
    console.print(f"  Not entailed   : {not_entailed}")
    console.print(f"  Error/timeout  : {errors}")

    # Yes/No accuracy
    yesno = [r for r in results if r["question_type"] == "yesno"]
    if yesno:
        def _gold_yn(r: dict) -> str:
            vals = [a[0].strip().lower() for a in r["gold_answers"]]
            return "yes" if "yes" in vals else "no"

        correct = sum(1 for r in yesno if r["predicted"] == _gold_yn(r))
        console.print(
            f"\n[bold]Yes/No questions[/bold] ({len(yesno)}): "
            f"{correct} / {len(yesno)} correct "
            f"({100 * correct // len(yesno)}%)"
        )

    # Value extraction
    value_q = [r for r in results if r["question_type"] == "value"]
    if value_q:
        got_val = sum(1 for r in value_q if r["entails"] and r["predicted_values"])
        true_no_val = sum(1 for r in value_q if r["entails"] and not r["predicted_values"])
        not_prov = sum(1 for r in value_q if r["entails"] is False)
        err = sum(1 for r in value_q if r["entails"] is None)
        console.print(
            f"\n[bold]Value questions[/bold] ({len(value_q)}): "
            f"value extracted: {got_val} | entailed (ground): {true_no_val} "
            f"| not provable: {not_prov} | error: {err}"
        )

    # Not-answerable accuracy
    na = [r for r in results if r["question_type"] == "not_answerable"]
    if na:
        # Correct when the hypothesis is not provable (entails=False)
        correct_na = sum(1 for r in na if r["entails"] is False)
        console.print(
            f"\n[bold]Not-answerable[/bold] ({len(na)}): "
            f"{correct_na} / {len(na)} correctly rejected"
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


@app.command()
def main(
    context_path: str = typer.Option(
        "results/all_context_with_logic.jsonl",
        "-c",
        "--context",
        help="Context KB file (all_context_with_logic.jsonl)",
    ),
    sq_path: str = typer.Option(
        "results/B100_question_with_logic.jsonl",
        "-s",
        "--sq",
        help="Question file (B100_question_with_logic.jsonl)",
    ),
    output_path: str = typer.Option(
        "results/B100_answers.jsonl",
        "-o",
        "--output",
        help="Output path for answer results",
    ),
    timeout: int = typer.Option(
        10,
        "-t",
        "--timeout",
        help="Per-query SWI-Prolog timeout in seconds",
    ),
) -> None:
    """Answer questions by running their extracted Prolog hypothesis against
    the merged context + scenario KB."""

    ctx_index = load_context_index(context_path)
    sq_records = load_json_or_jsonl(sq_path)

    console.print(f"Loaded {len(ctx_index)} context records (indexed by URL)")
    console.print(f"Loaded {len(sq_records)} question records")

    results: list[dict] = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Answering questions", total=len(sq_records))

        for sq_rec in sq_records:
            sq_data = sq_rec.get("data", {})
            url = sq_data.get("url", "")
            rec_id = sq_rec.get("id") or sq_data.get("id") or sq_rec.get("index")

            sq_prolog = extract_prolog_kb(sq_rec)
            ctx_rec = ctx_index.get(url)
            ctx_prolog = extract_prolog_kb(ctx_rec) if ctx_rec else None

            goal = sq_prolog.get("hypothesis", "").rstrip(".")
            q_type = _question_type(sq_data)

            program = build_prolog_program(ctx_prolog, sq_prolog)
            entails, values, error = query_prolog_bindings(program, goal, timeout=timeout)

            predicted = _predict(entails, values, q_type)

            results.append({
                "id": rec_id,
                "url": url,
                "question": sq_data.get("question"),
                "scenario": sq_data.get("scenario"),
                "question_type": q_type,
                "prolog_goal": goal,
                "context_found": ctx_rec is not None,
                "entails": entails,
                "predicted_values": values,
                "predicted": predicted,
                "gold_answers": sq_data.get("answers", []),
                "not_answerable": sq_data.get("not_answerable"),
                "error": error or None,
            })
            progress.advance(task)

    write_jsonl(output_path, results)
    console.print(f"Written {len(results)} answers to {output_path}")

    _print_summary(results)


if __name__ == "__main__":
    app()
