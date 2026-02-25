#!/usr/bin/env python3
"""Smoke-tests for the Prolog entailment checker.

Three test suites cover both supported hypothesis formats:

  Suite A – FOL (textual) path
      SQ records carry the hypothesis as a FOL string in ``hypothesis_fol``.
      ``check_single_record`` converts it to a Prolog goal automatically via
      ``fol_hypothesis_to_prolog`` before querying SWI-Prolog.
      Example: ``"Exists(t) CourtHearsBack(me, t)"``  →  ``court_hears_back(me, T)``

  Suite B – Extracted-Prolog path
      SQ records carry a ready-to-use Prolog goal string in
      ``logic_kb.prolog.hypothesis``.  ``check_single_record`` uses it directly,
      skipping FOL conversion entirely.
      Example: ``"can_claim_ctc(alice)"``

  Suite C – Full file pipeline
      Exercises the end-to-end ``run_entailment_check`` function (file I/O,
      context lookup, progress bar) using the same FOL records from Suite A.

Run:
    python scripts/test_entailment.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from symbolic_conqa.prolog_checker import check_single_record, run_entailment_check

# ---------------------------------------------------------------------------
# Shared context KBs
# ---------------------------------------------------------------------------

CONTEXT_RECORDS = [
    # Child Tax Credit rules
    {
        "index": 0,
        "id": None,
        "logic_kb": {
            "prolog": {
                "facts": [
                    "cannot_make_new_claim_ctc(general_public).",
                ],
                "rules": [
                    "can_claim_ctc(X) :- gets_wtc(X).",
                    "grandparent(X, Z) :- parent(X, Y), parent(Y, Z).",
                ],
                "optional_rules": [],
            },
        },
        "data": {
            "title": "Child Tax Credit",
            "url": "https://example.gov/child-tax-credit",
            "contents": [],
        },
    },
    # Special Guardianship rules
    {
        "index": 1,
        "id": None,
        "logic_kb": {
            "prolog": {
                "facts": [],
                "rules": [
                    "eligible_special_guardian(X) :- cares_for_child(X, _), \\+ parent(X, _).",
                ],
                "optional_rules": [],
            },
        },
        "data": {
            "title": "Special Guardianship",
            "url": "https://example.gov/special-guardian",
            "contents": [],
        },
    },
]

# Pre-built URL → record index for in-memory lookups
CTX_INDEX = {r["data"]["url"]: r for r in CONTEXT_RECORDS}

CTC_URL = "https://example.gov/child-tax-credit"
SG_URL  = "https://example.gov/special-guardian"

# ---------------------------------------------------------------------------
# Suite A – FOL (textual) hypothesis records
#
# The hypothesis is stored in ``logic_kb.hypothesis_fol`` as a FOL string.
# ``check_single_record`` calls ``fol_hypothesis_to_prolog`` to produce the
# Prolog goal at query time.
# ---------------------------------------------------------------------------

SQ_FOL = [
    # A-0  ENTAILS
    #   Context rule : can_claim_ctc(X) :- gets_wtc(X).
    #   Scenario fact: gets_wtc(alice).
    #   FOL hypothesis: CanClaimCtc(alice)  →  can_claim_ctc(alice)  →  True
    {
        "id": "fol-0",
        "logic_kb": {
            "prolog": {
                "facts": ["gets_wtc(alice)."],
                "rules": [],
                "optional_rules": [],
            },
            "hypothesis_fol": "CanClaimCtc(alice)",
        },
        "data": {
            "url": CTC_URL,
            "scenario": "Alice already receives Working Tax Credit.",
            "question": "Can Alice claim Child Tax Credit?",
            "not_answerable": False,
            "answers": [["yes", []]],
            "id": "fol-0",
        },
    },
    # A-1  NOT ENTAILS
    #   Bob has no WTC, so can_claim_ctc(bob) is not provable.
    #   FOL hypothesis: CanClaimCtc(bob)  →  can_claim_ctc(bob)  →  False
    {
        "id": "fol-1",
        "logic_kb": {
            "prolog": {
                "facts": ["employed(bob)."],
                "rules": [],
                "optional_rules": [],
            },
            "hypothesis_fol": "CanClaimCtc(bob)",
        },
        "data": {
            "url": CTC_URL,
            "scenario": "Bob is employed but does not get Working Tax Credit.",
            "question": "Can Bob claim Child Tax Credit?",
            "not_answerable": True,
            "answers": [["no", []]],
            "id": "fol-1",
        },
    },
    # A-2  CHAINED INFERENCE – ENTAILS
    #   Context rule: grandparent(X,Z) :- parent(X,Y), parent(Y,Z).
    #   Scenario    : parent(tom, bob). parent(bob, ann).
    #   FOL hypothesis: Grandparent(tom, ann)  →  grandparent(tom, ann)  →  True
    {
        "id": "fol-2",
        "logic_kb": {
            "prolog": {
                "facts": ["parent(tom, bob).", "parent(bob, ann)."],
                "rules": [],
                "optional_rules": [],
            },
            "hypothesis_fol": "Grandparent(tom, ann)",
        },
        "data": {
            "url": CTC_URL,
            "scenario": "Tom is Bob's parent. Bob is Ann's parent.",
            "question": "Is Tom Ann's grandparent?",
            "not_answerable": False,
            "answers": [["yes", []]],
            "id": "fol-2",
        },
    },
    # A-3  EXISTENTIAL QUANTIFIER – NO CONTEXT – NOT ENTAILS
    #   No context record matches this URL; predicate is undefined → fails.
    #   FOL hypothesis: Exists(t) CourtHearsBack(me, t)
    #                →  court_hears_back(me, T)  →  False
    {
        "id": "fol-3",
        "logic_kb": {
            "prolog": {
                "facts": ["applied(me)."],
                "rules": [],
                "optional_rules": [],
            },
            "hypothesis_fol": "Exists(t) CourtHearsBack(me, t)",
        },
        "data": {
            "url": "https://example.gov/unknown-page",
            "scenario": "I applied for something.",
            "question": "When will the court hear back?",
            "not_answerable": False,
            "answers": [["within 10 days", []]],
            "id": "fol-3",
        },
    },
]

EXPECTED_FOL = {
    "fol-0": True,   # WTC → can claim CTC
    "fol-1": False,  # no WTC → cannot claim CTC
    "fol-2": True,   # chained parent rules prove grandparent
    "fol-3": False,  # unknown URL, predicate undefined
}

# ---------------------------------------------------------------------------
# Suite B – Extracted-Prolog hypothesis records
#
# The hypothesis is stored directly in ``logic_kb.prolog.hypothesis`` as a
# ready-to-use Prolog goal string (snake_case, no quantifier prefix).
# ``check_single_record`` uses it as-is, with no FOL conversion.
# ---------------------------------------------------------------------------

SQ_PROLOG = [
    # B-0  ENTAILS  (mirrors A-0 scenario)
    #   Pre-extracted goal: "can_claim_ctc(alice)"
    {
        "id": "prolog-0",
        "logic_kb": {
            "prolog": {
                "facts": ["gets_wtc(alice)."],
                "rules": [],
                "optional_rules": [],
                "hypothesis": "can_claim_ctc(alice)",
            },
        },
        "data": {
            "url": CTC_URL,
            "scenario": "Alice already receives Working Tax Credit.",
            "question": "Can Alice claim Child Tax Credit?",
            "not_answerable": False,
            "answers": [["yes", []]],
            "id": "prolog-0",
        },
    },
    # B-1  NOT ENTAILS  (mirrors A-1 scenario)
    #   Pre-extracted goal: "can_claim_ctc(bob)"
    {
        "id": "prolog-1",
        "logic_kb": {
            "prolog": {
                "facts": ["employed(bob)."],
                "rules": [],
                "optional_rules": [],
                "hypothesis": "can_claim_ctc(bob)",
            },
        },
        "data": {
            "url": CTC_URL,
            "scenario": "Bob is employed but does not get Working Tax Credit.",
            "question": "Can Bob claim Child Tax Credit?",
            "not_answerable": True,
            "answers": [["no", []]],
            "id": "prolog-1",
        },
    },
    # B-2  CHAINED INFERENCE – ENTAILS  (mirrors A-2 scenario)
    #   Pre-extracted goal: "grandparent(tom, ann)"
    {
        "id": "prolog-2",
        "logic_kb": {
            "prolog": {
                "facts": ["parent(tom, bob).", "parent(bob, ann)."],
                "rules": [],
                "optional_rules": [],
                "hypothesis": "grandparent(tom, ann)",
            },
        },
        "data": {
            "url": CTC_URL,
            "scenario": "Tom is Bob's parent. Bob is Ann's parent.",
            "question": "Is Tom Ann's grandparent?",
            "not_answerable": False,
            "answers": [["yes", []]],
            "id": "prolog-2",
        },
    },
    # B-3  ENTAILS via special-guardian context
    #   Context rule: eligible_special_guardian(X) :- cares_for_child(X,_), \+ parent(X,_).
    #   Scenario    : carol cares for a child and is not the parent.
    #   Pre-extracted goal: "eligible_special_guardian(carol)"  →  True
    {
        "id": "prolog-3",
        "logic_kb": {
            "prolog": {
                "facts": ["cares_for_child(carol, child1)."],
                "rules": [],
                "optional_rules": [],
                "hypothesis": "eligible_special_guardian(carol)",
            },
        },
        "data": {
            "url": SG_URL,
            "scenario": "Carol has been caring for her nephew but is not his parent.",
            "question": "Is Carol eligible to apply as a special guardian?",
            "not_answerable": False,
            "answers": [["yes", []]],
            "id": "prolog-3",
        },
    },
    # B-4  NOT ENTAILS via special-guardian context
    #   Dave cares for a child but IS the parent → not eligible.
    #   Pre-extracted goal: "eligible_special_guardian(dave)"  →  False
    {
        "id": "prolog-4",
        "logic_kb": {
            "prolog": {
                "facts": ["cares_for_child(dave, child2).", "parent(dave, child2)."],
                "rules": [],
                "optional_rules": [],
                "hypothesis": "eligible_special_guardian(dave)",
            },
        },
        "data": {
            "url": SG_URL,
            "scenario": "Dave is caring for his own child.",
            "question": "Is Dave eligible to apply as a special guardian?",
            "not_answerable": True,
            "answers": [["no", []]],
            "id": "prolog-4",
        },
    },
]

EXPECTED_PROLOG = {
    "prolog-0": True,   # WTC → can claim CTC
    "prolog-1": False,  # no WTC → cannot claim CTC
    "prolog-2": True,   # chained parent rules prove grandparent
    "prolog-3": True,   # cares for child, not parent → eligible guardian
    "prolog-4": False,  # is parent → not eligible guardian
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_table(title: str, rows: list[tuple]) -> int:
    """Print a results table and return the number of passed tests."""
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")
    print(f"{'ID':<12} {'GOAL':<33} {'GOT':>7} {'EXP':>7} {'':>4}")
    print(f"{'-' * 65}")
    passed = 0
    for rid, goal, got, exp in rows:
        ok = "✓" if got == exp else "✗"
        if got == exp:
            passed += 1
        print(f"{rid!s:<12} {goal[:32]:<33} {str(got):>7} {str(exp):>7}  {ok}")
    print(f"{'=' * 65}")
    print(f"  Passed {passed}/{len(rows)}")
    return passed


# ---------------------------------------------------------------------------
# Suite A runner – FOL (textual) path via check_single_record
# ---------------------------------------------------------------------------


def run_fol_suite() -> int:
    """Check all FOL records using check_single_record (in-memory API)."""
    rows = []
    for sq_rec in SQ_FOL:
        url     = sq_rec["data"]["url"]
        ctx_rec = CTX_INDEX.get(url)              # None if URL not in context
        result  = check_single_record(sq_rec, ctx_rec)
        rows.append((result["id"], result["prolog_goal"], result["entails"],
                     EXPECTED_FOL.get(str(result["id"]))))
    return _print_table("Suite A – FOL / textual hypothesis  (check_single_record)", rows)


# ---------------------------------------------------------------------------
# Suite B runner – Extracted-Prolog path via check_single_record
# ---------------------------------------------------------------------------


def run_prolog_suite() -> int:
    """Check all extracted-Prolog records using check_single_record (in-memory API)."""
    rows = []
    for sq_rec in SQ_PROLOG:
        url     = sq_rec["data"]["url"]
        ctx_rec = CTX_INDEX.get(url)
        result  = check_single_record(sq_rec, ctx_rec)
        rows.append((result["id"], result["prolog_goal"], result["entails"],
                     EXPECTED_PROLOG.get(str(result["id"]))))
    return _print_table("Suite B – Extracted-Prolog hypothesis  (check_single_record)", rows)


# ---------------------------------------------------------------------------
# Suite C runner – full file pipeline via run_entailment_check
# ---------------------------------------------------------------------------


def run_pipeline_suite() -> int:
    """Exercise run_entailment_check end-to-end using the FOL records."""
    def write_jsonl(path: Path, records: list) -> None:
        with path.open("w") as f:
            for r in records:
                f.write(json.dumps(r, indent=4) + "\n\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        ctx_path = str(Path(tmpdir) / "context.jsonl")
        sq_path  = str(Path(tmpdir) / "sq.jsonl")
        out_path = str(Path(tmpdir) / "results.jsonl")

        write_jsonl(Path(ctx_path), CONTEXT_RECORDS)
        write_jsonl(Path(sq_path),  SQ_FOL)

        run_entailment_check(ctx_path, sq_path, out_path, timeout=10)

        with open(out_path) as f:
            results = [json.loads(b) for b in f.read().split("\n\n") if b.strip()]

    rows = [
        (r["id"], r["prolog_goal"], r["entails"], EXPECTED_FOL.get(str(r["id"])))
        for r in results
    ]
    return _print_table("Suite C – Full file pipeline  (run_entailment_check)", rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    total_passed = 0
    total_tests  = 0

    total_passed += run_fol_suite()
    total_tests  += len(SQ_FOL)

    total_passed += run_prolog_suite()
    total_tests  += len(SQ_PROLOG)

    total_passed += run_pipeline_suite()
    total_tests  += len(SQ_FOL)

    print(f"\n  Overall: {total_passed}/{total_tests} passed\n")


if __name__ == "__main__":
    main()
