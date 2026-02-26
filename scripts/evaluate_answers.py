#!/usr/bin/env python3
"""Evaluate Prolog-based answers in B100_answers.jsonl against ConditionalQA gold.

Metrics (official ConditionalQA evaluation):
  EM                  – exact-match of answer text after normalisation
  F1                  – token-level F1 of answer text
  EM_with_conditions  – EM weighted by condition F1
  F1_with_conditions  – F1 weighted by condition F1

Precision / Recall metrics (retrieval-style):
  Entailment P/R/F1   – treats entails=True as the positive prediction;
                        correct = EM==1.  Computed per question type.
  Yes/No P/R/F1       – binary classification where "yes" is the positive
                        class (yes/no subset only).

Results are broken down by question type (yes/no, value/extractive,
conditional) and by entailment outcome.

Usage:
    python scripts/evaluate_answers.py
    python scripts/evaluate_answers.py \\
        -p results/B100_answers.jsonl \\
        -r data/ConditionalQA/v1_0/dev.json
"""

# NOTE: Do NOT add `from __future__ import annotations` here.
# Typer inspects annotations at runtime.

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "data" / "ConditionalQA"))

import typer
from rich.console import Console
from rich.table import Table

from evaluate import compute_metrics, normalize_answer  # type: ignore[import]
from symbolic_conqa.io_utils import load_json_or_jsonl

console = Console()
app = typer.Typer(help="Evaluate Prolog answers against ConditionalQA gold.")


# ---------------------------------------------------------------------------
# Prediction format conversion
# ---------------------------------------------------------------------------


def to_conqa_answers(result: dict) -> list:
    """Convert a B100_answers result record to ConditionalQA answer format.

    ConditionalQA format: ``[[answer_text, [condition, ...]], ...]``
    Not-answerable questions use an empty list ``[]``.

    We do not predict conditions, so the condition list is always ``[]``.
    """
    pred = result["predicted"]

    # Errors, timeouts, and "not answerable" predictions → empty answer list
    if pred is None or pred == "not answerable":
        return []

    # Extracted variable bindings (list of Prolog atoms)
    if isinstance(pred, list):
        return [[v, []] for v in pred]

    # "yes" or "no"
    return [[pred, []]]


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------


def _avg(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def compute_all_metrics(
    results: list[dict],
    ref_index: dict[str, list],
) -> dict:
    """Run official compute_metrics on every result and collect per-type lists."""
    per_type: dict[str, dict[str, list]] = {
        "total":       {"em": [], "c_em": [], "f1": [], "c_f1": []},
        "yesno":       {"em": [], "c_em": [], "f1": [], "c_f1": []},
        "value":       {"em": [], "c_em": [], "f1": [], "c_f1": []},
        "not_answerable": {"em": [], "c_em": [], "f1": [], "c_f1": []},
        "conditional": {"em": [], "c_em": [], "f1": [], "c_f1": []},
    }

    for res in results:
        qid = str(res["id"])
        if qid not in ref_index:
            continue

        reference  = ref_index[qid]
        prediction = to_conqa_answers(res)
        em, c_em, f1, c_f1 = compute_metrics(prediction, reference)

        q_type = res["question_type"]   # yesno | value | not_answerable
        is_conditional = any(ans[1] for ans in reference) if reference else False

        for bucket in ["total", q_type]:
            per_type[bucket]["em"].append(em)
            per_type[bucket]["c_em"].append(c_em)
            per_type[bucket]["f1"].append(f1)
            per_type[bucket]["c_f1"].append(c_f1)

        if is_conditional:
            per_type["conditional"]["em"].append(em)
            per_type["conditional"]["c_em"].append(c_em)
            per_type["conditional"]["f1"].append(f1)
            per_type["conditional"]["c_f1"].append(c_f1)

    return {
        bucket: {
            "n":              len(d["em"]),
            "EM":             _avg(d["em"]),
            "EM_cond":        _avg(d["c_em"]),
            "F1":             _avg(d["f1"]),
            "F1_cond":        _avg(d["c_f1"]),
        }
        for bucket, d in per_type.items()
    }


# ---------------------------------------------------------------------------
# Precision / Recall
# ---------------------------------------------------------------------------


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Return (precision, recall, F1) from counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def compute_precision_recall(
    results: list[dict],
    ref_index: dict[str, list],
) -> dict:
    """Compute two flavours of precision/recall.

    **Entailment P/R** (all types):
      Positive prediction = ``entails is True``.
      Correct             = EM == 1.0.
      TP = entailed AND correct
      FP = entailed AND wrong
      FN = not entailed AND correct (missed a provable answer)

    **Yes/No binary P/R** (yes/no subset only):
      Positive class = gold answer is "yes".
      TP = predicted "yes" AND gold "yes"
      FP = predicted "yes" AND gold "no"
      FN = predicted "no"  AND gold "yes"
    """
    # Entailment P/R buckets
    ent: dict[str, dict[str, int]] = {
        k: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for k in ["total", "yesno", "value", "not_answerable"]
    }
    # Yes/No binary P/R
    yn = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    for res in results:
        qid = str(res["id"])
        ref  = ref_index.get(qid, [])
        pred = to_conqa_answers(res)
        em, *_ = compute_metrics(pred, ref)

        correct      = em == 1.0
        predicted_pos = res["entails"] is True
        qt = res["question_type"]

        for bucket in ["total", qt]:
            d = ent[bucket]
            if predicted_pos and correct:
                d["tp"] += 1
            elif predicted_pos and not correct:
                d["fp"] += 1
            elif not predicted_pos and correct:
                d["fn"] += 1
            else:
                d["tn"] += 1

        # Yes/No binary: "yes" is the positive class
        if qt == "yesno":
            gold_vals = {a[0].strip().lower() for a in ref}
            gold_yes  = "yes" in gold_vals
            pred_yes  = res["predicted"] == "yes"
            if pred_yes and gold_yes:
                yn["tp"] += 1
            elif pred_yes and not gold_yes:
                yn["fp"] += 1
            elif not pred_yes and gold_yes:
                yn["fn"] += 1
            else:
                yn["tn"] += 1

    def _row(d: dict) -> dict:
        p, r, f = _prf(d["tp"], d["fp"], d["fn"])
        return {"tp": d["tp"], "fp": d["fp"], "fn": d["fn"], "tn": d["tn"],
                "precision": p, "recall": r, "f1": f}

    return {
        "entailment": {k: _row(v) for k, v in ent.items()},
        "yesno_binary": _row(yn),
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def print_metrics_table(metrics: dict) -> None:
    table = Table(title="ConditionalQA Evaluation Results", show_lines=True)
    table.add_column("Subset",         style="bold")
    table.add_column("N",              justify="right")
    table.add_column("EM",             justify="right")
    table.add_column("EM+conditions",  justify="right")
    table.add_column("F1",             justify="right")
    table.add_column("F1+conditions",  justify="right")

    order = ["total", "yesno", "value", "not_answerable", "conditional"]
    labels = {
        "total":           "All",
        "yesno":           "Yes/No",
        "value":           "Value (extractive)",
        "not_answerable":  "Not answerable",
        "conditional":     "Conditional (subset)",
    }

    for key in order:
        m = metrics[key]
        if m["n"] == 0:
            continue
        table.add_row(
            labels[key],
            str(m["n"]),
            _pct(m["EM"]),
            _pct(m["EM_cond"]),
            _pct(m["F1"]),
            _pct(m["F1_cond"]),
        )

    console.print(table)


def print_precision_table(pr: dict) -> None:
    """Print entailment-based and yes/no binary precision/recall tables."""
    # --- Entailment P/R ---
    ent_table = Table(
        title="Precision / Recall  (entails=True as positive prediction, EM as correct)",
        show_lines=True,
    )
    ent_table.add_column("Subset",    style="bold")
    ent_table.add_column("TP",        justify="right")
    ent_table.add_column("FP",        justify="right")
    ent_table.add_column("FN",        justify="right")
    ent_table.add_column("TN",        justify="right")
    ent_table.add_column("Precision", justify="right")
    ent_table.add_column("Recall",    justify="right")
    ent_table.add_column("F1",        justify="right")

    labels = {
        "total":           "All",
        "yesno":           "Yes/No",
        "value":           "Value",
        "not_answerable":  "Not answerable",
    }
    for key, label in labels.items():
        m = pr["entailment"][key]
        ent_table.add_row(
            label,
            str(m["tp"]), str(m["fp"]), str(m["fn"]), str(m["tn"]),
            _pct(m["precision"]), _pct(m["recall"]), _pct(m["f1"]),
        )
    console.print(ent_table)

    # --- Yes/No binary P/R ---
    m = pr["yesno_binary"]
    yn_table = Table(
        title='Precision / Recall  (Yes/No subset — "yes" as positive class)',
        show_lines=True,
    )
    yn_table.add_column("TP", justify="right")
    yn_table.add_column("FP", justify="right")
    yn_table.add_column("FN", justify="right")
    yn_table.add_column("TN", justify="right")
    yn_table.add_column("Precision", justify="right")
    yn_table.add_column("Recall",    justify="right")
    yn_table.add_column("F1",        justify="right")
    yn_table.add_row(
        str(m["tp"]), str(m["fp"]), str(m["fn"]), str(m["tn"]),
        _pct(m["precision"]), _pct(m["recall"]), _pct(m["f1"]),
    )
    console.print(yn_table)


def print_entailment_breakdown(results: list[dict]) -> None:
    """Print a secondary table showing entailment outcome counts per question type."""
    rows: dict[str, dict[str, int]] = {}
    for r in results:
        qt = r["question_type"]
        if qt not in rows:
            rows[qt] = {"total": 0, "entailed": 0, "not_entailed": 0, "error": 0}
        rows[qt]["total"] += 1
        if r["entails"] is True:
            rows[qt]["entailed"] += 1
        elif r["entails"] is False:
            rows[qt]["not_entailed"] += 1
        else:
            rows[qt]["error"] += 1

    table = Table(title="Entailment Outcome Breakdown", show_lines=True)
    table.add_column("Question type", style="bold")
    table.add_column("Total",        justify="right")
    table.add_column("Entailed",     justify="right")
    table.add_column("Not entailed", justify="right")
    table.add_column("Error/timeout",justify="right")

    for qt, d in rows.items():
        table.add_row(qt, str(d["total"]), str(d["entailed"]),
                      str(d["not_entailed"]), str(d["error"]))

    console.print(table)


def print_error_sample(results: list[dict], ref_index: dict, n: int = 10) -> None:
    """Print a sample of incorrect yes/no predictions for inspection."""
    def _gold_yn(r: dict) -> str | None:
        for ans in r["gold_answers"]:
            if ans[0].lower() in ("yes", "no"):
                return ans[0].lower()
        return None

    wrong = [
        r for r in results
        if r["question_type"] == "yesno"
        and r["predicted"] != _gold_yn(r)
        and _gold_yn(r) is not None
    ][:n]

    if not wrong:
        return

    console.print(f"\n[bold]Sample incorrect Yes/No predictions[/bold] "
                  f"(showing up to {n})")
    for r in wrong:
        console.print(
            f"  [dim]{r['id']}[/dim]  "
            f"pred=[yellow]{r['predicted']}[/yellow]  "
            f"gold=[green]{_gold_yn(r)}[/green]  "
            f"  {r['question'][:70]}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@app.command()
def main(
    pred_path: str = typer.Option(
        "results/B100_answers.jsonl",
        "-p",
        "--pred",
        help="Prolog answer results file",
    ),
    ref_path: str = typer.Option(
        "data/ConditionalQA/v1_0/dev.json",
        "-r",
        "--ref",
        help="ConditionalQA gold reference file (dev.json)",
    ),
    output_path: str = typer.Option(
        "",
        "-o",
        "--output",
        help="Optional path to write per-question metrics as JSON",
    ),
    error_sample: int = typer.Option(
        10,
        "--errors",
        help="Number of wrong yes/no predictions to show (0 to suppress)",
    ),
) -> None:
    """Evaluate Prolog-based answers against the ConditionalQA gold standard."""

    # Load predictions
    results = load_json_or_jsonl(pred_path)
    console.print(f"Loaded {len(results)} predictions from [cyan]{pred_path}[/cyan]")

    # Load reference and index by id
    ref_data = json.load(open(ref_path, encoding="utf-8"))
    ref_index: dict[str, list] = {str(r["id"]): r["answers"] for r in ref_data}
    console.print(f"Loaded {len(ref_index)} references from [cyan]{ref_path}[/cyan]")

    # Check coverage
    pred_ids = {str(r["id"]) for r in results}
    matched  = pred_ids & ref_index.keys()
    missing  = pred_ids - ref_index.keys()
    console.print(f"Coverage: {len(matched)}/{len(pred_ids)} predictions matched to gold")
    if missing:
        console.print(f"[yellow]Warning: {len(missing)} prediction IDs not in reference[/yellow]")

    # Compute metrics
    metrics = compute_all_metrics(results, ref_index)
    pr      = compute_precision_recall(results, ref_index)

    # Display
    console.print()
    print_metrics_table(metrics)
    console.print()
    print_precision_table(pr)
    console.print()
    print_entailment_breakdown(results)

    if error_sample > 0:
        print_error_sample(results, ref_index, n=error_sample)

    # Optionally write per-question metrics to JSON
    if output_path:
        per_question = []
        for res in results:
            qid = str(res["id"])
            ref = ref_index.get(qid, [])
            pred = to_conqa_answers(res)
            em, c_em, f1, c_f1 = compute_metrics(pred, ref)
            per_question.append({
                "id": res["id"],
                "question_type": res["question_type"],
                "question": res["question"],
                "prolog_goal": res["prolog_goal"],
                "predicted": res["predicted"],
                "gold_answers": res["gold_answers"],
                "entails": res["entails"],
                "correct": em == 1.0,
                "precision": 1.0 if (res["entails"] and em == 1.0) else
                             0.0 if res["entails"] else None,
                "em": em, "em_cond": c_em, "f1": f1, "f1_cond": c_f1,
            })
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(per_question, f, ensure_ascii=False, indent=2)
        console.print(f"\nPer-question metrics written to [cyan]{output_path}[/cyan]")


if __name__ == "__main__":
    app()
