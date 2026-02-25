#!/usr/bin/env python3
"""Quick smoke-test for the Prolog entailment checker.

Creates three synthetic (context, SQ) pairs with known expected outcomes:
  1. ENTAILS     – SQ facts + context rules prove the hypothesis.
  2. NOT ENTAILS – hypothesis contradicts / is unprovable from the KB.
  3. NO CONTEXT  – no matching context URL; hypothesis checked on SQ facts only.

Run:
    python scripts/test_entailment.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from symbolic_conqa.prolog_checker import run_entailment_check

# ---------------------------------------------------------------------------
# Synthetic context records
# ---------------------------------------------------------------------------

CONTEXT_RECORDS = [
    # --- Child Tax Credit rules ---
    {
        "index": 0,
        "id": None,
        "logic_kb": {
            "constants": [],
            "predicates": [],
            "fol": {"facts": [], "rules": [], "optional_rules": []},
            "prolog": {
                "facts": [
                    "cannot_make_new_claim_ctc(general_public).",
                ],
                "rules": [
                    # can claim CTC only if already on WTC
                    "can_claim_ctc(X) :- gets_wtc(X).",
                    # grandparent rule for testing chained inference
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
    # --- Special Guardian rules ---
    {
        "index": 1,
        "id": None,
        "logic_kb": {
            "constants": [],
            "predicates": [],
            "fol": {"facts": [], "rules": [], "optional_rules": []},
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

# ---------------------------------------------------------------------------
# Synthetic SQ records
# ---------------------------------------------------------------------------

SQ_RECORDS = [
    # 1. SHOULD ENTAIL
    #    Context rule: can_claim_ctc(X) :- gets_wtc(X).
    #    Scenario fact: gets_wtc(alice).
    #    Hypothesis:    can_claim_ctc(alice)   → True
    {
        "index": 0,
        "id": "test-0",
        "logic_kb": {
            "constants": [],
            "predicates": [],
            "fol": {"facts": [], "rules": [], "optional_rules": []},
            "prolog": {
                "facts": ["gets_wtc(alice)."],
                "rules": [],
                "optional_rules": [],
            },
            "hypothesis_fol": "CanClaimCtc(alice)",
        },
        "data": {
            "url": "https://example.gov/child-tax-credit",
            "scenario": "Alice already gets Working Tax Credit.",
            "question": "Can Alice claim Child Tax Credit?",
            "not_answerable": False,
            "answers": [["yes", []]],
            "evidences": [],
            "id": "test-0",
        },
    },
    # 2. SHOULD NOT ENTAIL
    #    Context rule: can_claim_ctc(X) :- gets_wtc(X).
    #    Scenario fact: does NOT assert gets_wtc(bob).
    #    Hypothesis:    can_claim_ctc(bob)     → False
    {
        "index": 1,
        "id": "test-1",
        "logic_kb": {
            "constants": [],
            "predicates": [],
            "fol": {"facts": [], "rules": [], "optional_rules": []},
            "prolog": {
                "facts": ["employed(bob)."],   # irrelevant fact; no WTC
                "rules": [],
                "optional_rules": [],
            },
            "hypothesis_fol": "CanClaimCtc(bob)",
        },
        "data": {
            "url": "https://example.gov/child-tax-credit",
            "scenario": "Bob is employed but does not get Working Tax Credit.",
            "question": "Can Bob claim Child Tax Credit?",
            "not_answerable": True,
            "answers": [["no", []]],
            "evidences": [],
            "id": "test-1",
        },
    },
    # 3. CHAINED INFERENCE – SHOULD ENTAIL
    #    Context rules: grandparent(X,Z) :- parent(X,Y), parent(Y,Z).
    #    Scenario facts: parent(tom, bob). parent(bob, ann).
    #    Hypothesis:     grandparent(tom, ann) → True
    {
        "index": 2,
        "id": "test-2",
        "logic_kb": {
            "constants": [],
            "predicates": [],
            "fol": {"facts": [], "rules": [], "optional_rules": []},
            "prolog": {
                "facts": ["parent(tom, bob).", "parent(bob, ann)."],
                "rules": [],
                "optional_rules": [],
            },
            "hypothesis_fol": "Grandparent(tom, ann)",
        },
        "data": {
            "url": "https://example.gov/child-tax-credit",
            "scenario": "Tom is Bob's parent. Bob is Ann's parent.",
            "question": "Is Tom Ann's grandparent?",
            "not_answerable": False,
            "answers": [["yes", []]],
            "evidences": [],
            "id": "test-2",
        },
    },
    # 4. NO CONTEXT MATCH – unprovable hypothesis
    #    URL does not appear in any context record.
    {
        "index": 3,
        "id": "test-3",
        "logic_kb": {
            "constants": [],
            "predicates": [],
            "fol": {"facts": [], "rules": [], "optional_rules": []},
            "prolog": {
                "facts": ["applied(me)."],
                "rules": [],
                "optional_rules": [],
            },
            "hypothesis_fol": "Exists(t) CourtHearsBack(me, t)",
        },
        "data": {
            "url": "https://example.gov/some-other-page",
            "scenario": "I applied for something.",
            "question": "When will the court hear back?",
            "not_answerable": False,
            "answers": [["within 10 days", []]],
            "evidences": [],
            "id": "test-3",
        },
    },
]

# ---------------------------------------------------------------------------
# Expected outcomes
# ---------------------------------------------------------------------------

EXPECTED = {
    "test-0": True,   # entails
    "test-1": False,  # does not entail
    "test-2": True,   # chained inference entails
    "test-3": False,  # no context → predicate undefined → False
}


def main() -> None:
    # Write temp JSONL files
    def write_jsonl(path: Path, records: list) -> None:
        with path.open("w") as f:
            for r in records:
                f.write(json.dumps(r, indent=4) + "\n\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        ctx_path = str(Path(tmpdir) / "context.jsonl")
        sq_path  = str(Path(tmpdir) / "sq.jsonl")
        out_path = str(Path(tmpdir) / "results.jsonl")

        write_jsonl(Path(ctx_path), CONTEXT_RECORDS)
        write_jsonl(Path(sq_path),  SQ_RECORDS)

        run_entailment_check(ctx_path, sq_path, out_path, timeout=10)

        # Load and evaluate results
        with open(out_path) as f:
            results = [json.loads(b) for b in f.read().split("\n\n") if b.strip()]

    print("\n" + "=" * 60)
    print(f"{'ID':<10} {'GOAL':<35} {'GOT':>8} {'EXP':>8} {'PASS':>6}")
    print("=" * 60)

    passed = 0
    for r in results:
        rid = r["id"]
        goal = r["prolog_goal"][:34]
        got  = r["entails"]
        exp  = EXPECTED.get(str(rid))
        ok   = "✓" if got == exp else "✗"
        if got == exp:
            passed += 1
        print(f"{rid!s:<10} {goal:<35} {str(got):>8} {str(exp):>8} {ok:>6}")

    print("=" * 60)
    print(f"Passed {passed}/{len(results)}\n")


if __name__ == "__main__":
    main()
