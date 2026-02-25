"""SWI-Prolog entailment checker for context + scenario/question logic KBs."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from .io_utils import load_json_or_jsonl

console = Console()


# ---------------------------------------------------------------------------
# FOL → Prolog goal conversion
# ---------------------------------------------------------------------------


def _camel_to_snake(name: str) -> str:
    """Convert a CamelCase predicate name to snake_case."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def fol_hypothesis_to_prolog(hypothesis_fol: str) -> str:
    """Convert a simple FOL hypothesis string to a Prolog goal string.

    Handles the following patterns:
      "Exists(t) CourtHearsBack(me, t)"      -> "court_hears_back(me, T)"
      "Exists(x) AdopterAgeReq(x, uk)"       -> "adopter_age_req(X, uk)"
      "CanClaimChildTaxCredit(you)"           -> "can_claim_child_tax_credit(you)"

    Variables introduced by quantifiers (Exists/Forall) are uppercased to
    become Prolog logical variables.  All other tokens are left as-is
    (constants stay lowercase).
    """
    h = hypothesis_fol.strip()
    if not h:
        return ""

    # Collect variable names from quantifier prefixes: Exists(x), Forall(x),
    # forall x, exists x  — require a non-word char (or start) before the
    # keyword so we don't match "Exists" inside names like "AdopterAgeRequirementExists".
    variables: set[str] = set()
    for m in re.finditer(
        r"(?<!\w)(?:Exists|Forall|forall|exists)\s*\(?\s*(\w+)\s*\)?", h, re.IGNORECASE
    ):
        variables.add(m.group(1).lower())

    # Strip all quantifier prefixes (same guard: negative lookbehind)
    h = re.sub(
        r"(?<!\w)(?:Exists|Forall|forall|exists)\s*\(?\s*\w+\s*\)?\s*",
        "",
        h,
        flags=re.IGNORECASE,
    ).strip()

    # Match: PredName(arg1, arg2, ...) — possibly no args
    m = re.match(r"([A-Za-z_]\w*)\s*(?:\((.+)\))?\s*$", h, re.DOTALL)
    if not m:
        return _camel_to_snake(h)

    pred = _camel_to_snake(m.group(1))
    raw_args = m.group(2)

    if raw_args is None:
        return pred  # zero-arity predicate

    # Simple comma split (safe for our flat argument lists)
    args = [a.strip() for a in raw_args.split(",")]

    prolog_args: list[str] = []
    for arg in args:
        if arg.lower() in variables:
            # Quantified variable → Prolog uppercase variable
            prolog_args.append(arg.upper())
        else:
            prolog_args.append(arg)

    return f"{pred}({', '.join(prolog_args)})"


# ---------------------------------------------------------------------------
# Prolog program builder
# ---------------------------------------------------------------------------


def build_prolog_program(
    context_prolog: dict[str, Any] | None,
    sq_prolog: dict[str, Any],
) -> str:
    """Combine context and SQ Prolog blocks into a single program string."""

    def _ensure_dot(clause: str) -> str:
        c = clause.strip()
        return c if c.endswith(".") else c + "."

    lines: list[str] = [
        ":- set_prolog_flag(unknown, fail).",  # undefined predicates fail, not error
        ":- use_module(library(lists)).",
    ]

    def _add_block(block: dict[str, Any], label: str) -> None:
        lines.append(f"\n% --- {label} ---")
        for clause in block.get("facts", []) + block.get("rules", []):
            if clause.strip():
                lines.append(_ensure_dot(clause))
        for clause in block.get("optional_rules", []):
            if clause.strip():
                lines.append(f"% optional: {_ensure_dot(clause)}")

    if context_prolog:
        _add_block(context_prolog, "context KB")
    _add_block(sq_prolog, "scenario KB")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SWI-Prolog subprocess runner
# ---------------------------------------------------------------------------


def check_entailment_swipl(
    program: str,
    goal: str,
    timeout: int = 10,
) -> tuple[bool | None, str]:
    """Check whether *program* (a Prolog program string) entails *goal*.

    The goal is embedded as an initialization directive inside a temporary
    file, avoiding shell-quoting issues entirely.

    Returns:
        (result, error_message)
        result = True   → goal provable
               = False  → goal not provable
               = None   → timeout or Prolog error
    """
    if not goal:
        return None, "empty goal"

    # Embed the query as an initialization directive.
    # We bind R to 0/1/2 first, then call halt(R) outside the catch so that
    # the halt/1 exception itself is never swallowed by catch/3.
    full_program = (
        program
        + f"\n\n:- catch(({goal} -> R=0 ; R=1), _, R=2), halt(R).\n"
    )

    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".pl")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(full_program)

        result = subprocess.run(
            ["swipl", "-q", tmp_name],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return True, ""
        elif result.returncode == 1:
            return False, ""
        elif result.returncode == 2:
            return None, "prolog exception during query"
        else:
            stderr = result.stderr.strip()
            return None, f"swipl error (rc={result.returncode}): {stderr[:300]}"

    except subprocess.TimeoutExpired:
        return None, "timeout"
    except FileNotFoundError:
        return None, "swipl not found — install with: sudo apt install swi-prolog"
    finally:
        os.unlink(tmp_name)


# ---------------------------------------------------------------------------
# Context index
# ---------------------------------------------------------------------------


def load_context_index(path: str) -> dict[str, dict[str, Any]]:
    """Load context records and index them by their ``data.url`` field."""
    records = load_json_or_jsonl(path)
    index: dict[str, dict[str, Any]] = {}
    for rec in records:
        url = rec.get("data", {}).get("url")
        if url:
            index[url] = rec
    return index


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_entailment_check(
    context_path: str,
    sq_path: str,
    output_path: str,
    timeout: int = 10,
) -> None:
    """Check Prolog entailment for every SQ record against its context KB.

    For each SQ record:
      1. Look up the matching context KB by URL.
      2. Merge context + scenario Prolog programs.
      3. Derive a Prolog goal from the hypothesis (prolog.hypothesis preferred,
         then auto-converted from hypothesis_fol).
      4. Run SWI-Prolog and record the entailment result.
      5. Write results to *output_path* (indented JSONL).
    """
    context_index = load_context_index(context_path)
    sq_records = load_json_or_jsonl(sq_path)

    console.print(f"Loaded {len(context_index)} context records (indexed by URL)")
    console.print(f"Loaded {len(sq_records)} SQ records")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Checking entailment", total=len(sq_records))

        for sq_rec in sq_records:
            sq_data = sq_rec.get("data", {})
            url = sq_data.get("url", "")
            rec_id = sq_rec.get("id") or sq_data.get("id") or sq_rec.get("index")

            sq_kb = sq_rec.get("logic_kb", {})
            sq_prolog = sq_kb.get("prolog", {})

            # Hypothesis: prefer an explicit Prolog hypothesis; fall back to
            # auto-converting the stored hypothesis_fol string.
            prolog_hypothesis: str = sq_kb.get("prolog", {}).get("hypothesis") or ""
            fol_hypothesis: str = (
                sq_kb.get("hypothesis_fol")
                or sq_kb.get("fol", {}).get("hypothesis")
                or ""
            )

            if prolog_hypothesis:
                goal = prolog_hypothesis.rstrip(".")
            elif fol_hypothesis:
                goal = fol_hypothesis_to_prolog(fol_hypothesis)
            else:
                goal = ""

            # Look up the context KB for this URL
            ctx_rec = context_index.get(url)
            ctx_prolog: dict[str, Any] | None = (
                ctx_rec["logic_kb"]["prolog"] if ctx_rec else None
            )
            context_found = ctx_rec is not None

            # Build merged Prolog program and run the check
            program = build_prolog_program(ctx_prolog, sq_prolog)
            entails, error = check_entailment_swipl(program, goal, timeout=timeout)

            result: dict[str, Any] = {
                "id": rec_id,
                "url": url,
                "context_found": context_found,
                "hypothesis_fol": fol_hypothesis,
                "prolog_goal": goal,
                "entails": entails,
                "not_answerable": sq_data.get("not_answerable"),
                "gold_answers": sq_data.get("answers", []),
                "error": error or None,
            }
            results.append(result)
            progress.advance(task)

    with out_file.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, indent=4) + "\n\n")

    console.print(f"Written {len(results)} results to {output_path}")

    # Summary statistics
    entailed = sum(1 for r in results if r["entails"] is True)
    not_entailed = sum(1 for r in results if r["entails"] is False)
    errors = sum(1 for r in results if r["entails"] is None)
    no_ctx = sum(1 for r in results if not r["context_found"])
    console.print(
        f"  Entailed: {entailed} | Not entailed: {not_entailed} "
        f"| Errors/skipped: {errors} | No context match: {no_ctx}"
    )
