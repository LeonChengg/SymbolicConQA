#!/usr/bin/env python3
"""Answer questions in B100_question_with_logic.jsonl using context KB from
all_context_with_logic.jsonl via SWI-Prolog entailment.

For each question record the script:
  1. Looks up the matching context KB by ``data.url``.
  2. Merges context facts/rules with the scenario facts/rules.
  3. Optionally applies alignment strategies to bridge predicate name mismatches.
  4. Uses the pre-extracted ``logic_kb.prolog.hypothesis`` as the Prolog goal.
  5. Runs the goal against SWI-Prolog:
       - Ground goals  (no free variables): True → "yes", False → "no"
       - Variable goals (e.g. ``time_to_hear_back(me, T)``):
         all solutions for the free variable(s) are collected and returned.
  6. Writes results to an output JSONL file and prints a summary.

Requires SWI-Prolog:
    sudo apt install swi-prolog   # Debian/Ubuntu
    conda install -c conda-forge swi-prolog

Usage:
    python scripts/answer_questions.py
    python scripts/answer_questions.py \\
        -c results/all_context_with_logic.jsonl \\
        -s results/B100_question_with_logic.jsonl \\
        -o results/B100_answers.jsonl

Alignment flags (optional):
    --normalize                          Enable stop-word normalisation bridging
    --auto-bridge                        Enable fuzzy token-overlap auto-bridging
    --auto-bridge-threshold 0.25         Jaccard threshold for auto-bridge
    --alias can_apply=can_become         Explicit predicate alias (repeatable)
    --bridge-rule "h(X):-c(X)."         Verbatim Prolog bridge clause (repeatable)
    --semantic-bridge                    Enable WordNet hyponym/hypernym bridging
    --semantic-bridge-threshold 0.3      Minimum WordNet entailment score
    --semantic-bridge-max-depth 4        Max hypernym path depth
"""

# NOTE: Do NOT add `from __future__ import annotations` here.
# Typer inspects annotations at runtime.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from symbolic_conqa.alignment import AlignmentConfig, align
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
        "results/all_SQ_with_logic_yesno.jsonl",
        "-s",
        "--sq",
        help="Question file (all_SQ_with_logic_yesno.jsonl)",
    ),
    output_path: str = typer.Option(
        "results/all_answers.jsonl",
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
    # ---- alignment flags ----
    normalize: bool = typer.Option(
        False,
        "--normalize/--no-normalize",
        help="Enable stop-word normalisation alignment strategy",
    ),
    auto_bridge: bool = typer.Option(
        False,
        "--auto-bridge/--no-auto-bridge",
        help="Enable fuzzy token-overlap auto-bridge strategy",
    ),
    auto_bridge_threshold: float = typer.Option(
        0.2,
        "--auto-bridge-threshold",
        help="Jaccard threshold for auto-bridge (default 0.2)",
    ),
    aliases: list[str] = typer.Option(
        [],
        "--alias",
        help=(
            "Explicit predicate alias as 'hyp_pred=ctx_pred'. "
            "Repeat for multiple aliases."
        ),
    ),
    bridge_rules: list[str] = typer.Option(
        [],
        "--bridge-rule",
        help=(
            "Verbatim Prolog bridge clause, e.g. 'h(X):-c(X).'. "
            "Repeat for multiple rules."
        ),
    ),
    semantic_bridge: bool = typer.Option(
        False,
        "--semantic-bridge/--no-semantic-bridge",
        help=(
            "Enable WordNet hyponym/hypernym semantic bridge strategy. "
            "Requires NLTK with WordNet corpus "
            "('python -m nltk.downloader wordnet')."
        ),
    ),
    semantic_bridge_threshold: float = typer.Option(
        0.3,
        "--semantic-bridge-threshold",
        help="Minimum WordNet entailment score to accept a candidate (default 0.3)",
    ),
    semantic_bridge_max_depth: int = typer.Option(
        4,
        "--semantic-bridge-max-depth",
        help="Maximum hypernym path depth for semantic bridge (default 4)",
    ),
    constant_align: bool = typer.Option(
        False,
        "--constant-align/--no-constant-align",
        help="Variabilize person-typed constants for alignment",
    ),
) -> None:
    """Answer questions by running their extracted Prolog hypothesis against
    the merged context + scenario KB, with optional alignment strategies."""

    # Parse alias strings "hyp=ctx" → dict
    alias_dict: dict[str, str] = {}
    for alias_str in aliases:
        if "=" not in alias_str:
            console.print(
                f"[yellow]Warning:[/yellow] --alias '{alias_str}' ignored "
                "(expected 'hyp_pred=ctx_pred' format)"
            )
            continue
        hyp_pred, ctx_pred = alias_str.split("=", 1)
        alias_dict[hyp_pred.strip()] = ctx_pred.strip()

    alignment_cfg = AlignmentConfig(
        normalize=normalize,
        auto_bridge=auto_bridge,
        auto_bridge_threshold=auto_bridge_threshold,
        aliases=alias_dict,
        bridge_rules=list(bridge_rules),
        semantic_bridge=semantic_bridge,
        semantic_bridge_threshold=semantic_bridge_threshold,
        semantic_bridge_max_depth=semantic_bridge_max_depth,
        constant_align=constant_align,
    )

    use_alignment = (
        normalize or auto_bridge or alias_dict or bridge_rules
        or semantic_bridge or constant_align
    )
    if use_alignment:
        strategies = []
        if normalize:
            strategies.append("normalize")
        if alias_dict:
            strategies.append(f"aliases({len(alias_dict)})")
        if bridge_rules:
            strategies.append(f"bridge_rules({len(bridge_rules)})")
        if auto_bridge:
            strategies.append(f"auto_bridge(threshold={auto_bridge_threshold})")
        if semantic_bridge:
            strategies.append(
                f"semantic_bridge(threshold={semantic_bridge_threshold}, "
                f"max_depth={semantic_bridge_max_depth})"
            )
        if constant_align:
            strategies.append("constant_align")
        console.print(f"Alignment strategies: [cyan]{', '.join(strategies)}[/cyan]")

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

            if use_alignment:
                ctx_kb_full = ctx_rec.get("logic_kb", {}) if ctx_rec else None
                sq_kb_full = sq_rec.get("logic_kb", {})
                program, goal = align(
                    program, goal, ctx_prolog, sq_prolog, alignment_cfg,
                    ctx_kb=ctx_kb_full, sq_kb=sq_kb_full,
                )

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
