"""Alignment layer for bridging hypothesis predicates to context KB predicates.

When the LLM extracts logic from context and questions separately, it often
chooses different predicate names for the same concept — e.g. the hypothesis
``time_to_hear_back_from_court(me, T)`` vs the context fact
``court_sends_case_number_within_10_days(court, applicant)``.

This module provides five composable strategies that inject Prolog bridge rules
to connect hypothesis predicates to their context counterparts, **without
modifying the original KB**.

Strategies (all optional, applied in order):

  1. ``normalize``       – Strip common English stop-words from predicate token
                           lists; where the normalised forms of a hypothesis
                           predicate and a context predicate match, inject a
                           bridge rule.

  2. ``aliases``         – Explicit ``{hyp_pred: ctx_pred}`` mapping supplied
                           by the caller.  Each entry becomes a Prolog bridge
                           rule.

  3. ``bridge_rules``    – Verbatim Prolog clauses provided by the caller,
                           injected as-is.  Use this for precise hand-crafted
                           bridges that the automatic strategies cannot
                           generate.

  4. ``auto_bridge``     – Fuzzy token-overlap search: for every predicate that
                           appears in the hypothesis goal and is not already
                           defined in the program, find the best-matching
                           context predicate and generate a forwarding bridge
                           rule.

  5. ``semantic_bridge`` – WordNet-based directional semantic bridge: for every
                           undefined hypothesis predicate, search for context
                           predicates whose tokens are **hyponyms** (more
                           specific concepts) of the hypothesis tokens.  A
                           bridge rule ``hyp :- ctx`` is then logically sound
                           because the more-specific fact entails the
                           more-general goal.

                           Example: hypothesis goal ``die_in(P, L)`` vs
                           context fact ``killed_in(P, L)``.  WordNet tells us
                           "kill" is a hyponym of "die" (being killed entails
                           dying), so the bridge ``die_in(_A0,_A1) :-
                           killed_in(_A0,_A1).`` is injected automatically.

                           Requires ``nltk`` with the WordNet corpus
                           (``python -m nltk.downloader wordnet``).

Usage::

    from symbolic_conqa.alignment import AlignmentConfig, align

    config = AlignmentConfig(
        auto_bridge=True,
        auto_bridge_threshold=0.25,
        semantic_bridge=True,          # WordNet hyponym/hypernym bridging
        semantic_bridge_threshold=0.3,
        aliases={"can_apply_to_change_surname": "can_change_surname_with_consent"},
        bridge_rules=[
            "time_to_hear_back_from_court(_, within_10_days) "
            ":- court_sends_case_number_within_10_days(_, _).",
        ],
    )

    program, goal = align(program, goal, ctx_prolog, sq_prolog, config)
    entails, values, error = query_prolog_bindings(program, goal)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Default stop-word set for predicate normalisation
# ---------------------------------------------------------------------------

_DEFAULT_STOP: frozenset[str] = frozenset(
    {
        # articles / determiners
        "a", "an", "the", "this", "that",
        # copula / auxiliaries
        "is", "are", "was", "were", "be", "been", "being",
        "has", "have", "had", "do", "does", "did",
        "can", "could", "will", "would", "shall", "should", "may", "might",
        # common prepositions
        "to", "from", "with", "for", "in", "of", "at", "by", "on",
        "into", "about", "over", "under", "after", "before",
        # conjunctions / discourse
        "and", "or", "not", "if", "then", "else",
        # generic QA tokens
        "get", "give", "make", "take",
    }
)

# Regex that matches a lowercase predicate name at a word boundary
_PRED_NAME_RE = re.compile(r"\b([a-z][a-z0-9_]{1,})\s*\(")
# Matches the head of a Prolog clause (before :- or .)
_CLAUSE_HEAD_RE = re.compile(
    r"^\s*([a-z][a-z0-9_]*)\s*(?:\(([^)]*)\))?\s*(?::-|\.)", re.MULTILINE
)


# ---------------------------------------------------------------------------
# Internal helpers: predicate extraction
# ---------------------------------------------------------------------------


def _top_level_comma_count(args: str) -> int:
    """Count top-level commas (ignoring nested parentheses)."""
    depth, count = 0, 0
    for ch in args:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            count += 1
    return count


def _extract_defined_predicates(program: str) -> dict[str, int]:
    """Return {predicate_name: arity} for all heads defined in *program*."""
    preds: dict[str, int] = {}
    for m in _CLAUSE_HEAD_RE.finditer(program):
        name = m.group(1)
        args_str = m.group(2) or ""
        arity = (_top_level_comma_count(args_str) + 1) if args_str.strip() else 0
        preds[name] = arity
    return preds


def _extract_preds_from_block(block: dict[str, Any]) -> dict[str, int]:
    """Return {name: arity} for all predicate heads in a prolog KB block."""
    preds: dict[str, int] = {}
    for clause in block.get("facts", []) + block.get("rules", []):
        head = clause.split(":-")[0].strip().rstrip(".")
        m = re.match(r"([a-z][a-z0-9_]*)\s*\(([^)]*)\)", head)
        if m:
            name = m.group(1)
            args_str = m.group(2)
            arity = (_top_level_comma_count(args_str) + 1) if args_str.strip() else 0
            preds[name] = arity
        else:
            m0 = re.match(r"([a-z][a-z0-9_]*)$", head)
            if m0:
                preds[m0.group(1)] = 0
    return preds


def _extract_preds_from_goal(goal: str) -> dict[str, int]:
    """Return {name: arity} for all predicate calls in *goal*."""
    preds: dict[str, int] = {}
    for m in re.finditer(r"([a-z][a-z0-9_]*)\s*\(([^)]*)\)", goal):
        name = m.group(1)
        args_str = m.group(2)
        arity = (_top_level_comma_count(args_str) + 1) if args_str.strip() else 0
        preds[name] = arity
    return preds


# ---------------------------------------------------------------------------
# Strategy 1 — Normalisation
# ---------------------------------------------------------------------------


def normalize_predicate(name: str, stop: frozenset[str] = _DEFAULT_STOP) -> str:
    """Return the normalised form of *name* by dropping stop-word tokens.

    Example::

        normalize_predicate("time_to_hear_back_from_court")
        # → "time_hear_back_court"

        normalize_predicate("can_apply_special_guardian")
        # → "apply_special_guardian"
    """
    tokens = name.split("_")
    kept = [t for t in tokens if t and t.lower() not in stop]
    return "_".join(kept) if kept else name


def find_normalized_matches(
    hyp_preds: dict[str, int],
    ctx_preds: dict[str, int],
    stop: frozenset[str] = _DEFAULT_STOP,
) -> list[tuple[str, str, int, int]]:
    """Return ``(hyp_pred, ctx_pred, hyp_arity, ctx_arity)`` pairs where
    the normalised forms are identical.
    """
    ctx_norm: dict[str, tuple[str, int]] = {
        normalize_predicate(p, stop): (p, a) for p, a in ctx_preds.items()
    }
    matches = []
    for hyp_pred, hyp_arity in hyp_preds.items():
        norm = normalize_predicate(hyp_pred, stop)
        if norm in ctx_norm:
            ctx_pred, ctx_arity = ctx_norm[norm]
            if hyp_pred != ctx_pred:  # only bridge if names actually differ
                matches.append((hyp_pred, ctx_pred, hyp_arity, ctx_arity))
    return matches


# ---------------------------------------------------------------------------
# Strategy 4 — Auto-bridge via token-overlap (Jaccard)
# ---------------------------------------------------------------------------


def token_jaccard(a: str, b: str) -> float:
    """Jaccard similarity between the token sets of two predicate names."""
    ta = set(a.split("_"))
    tb = set(b.split("_"))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def find_best_ctx_match(
    hyp_pred: str,
    ctx_preds: dict[str, int],
    threshold: float = 0.2,
    top_k: int = 1,
) -> list[tuple[str, int, float]]:
    """Return up to *top_k* ``(ctx_pred, arity, score)`` tuples with
    Jaccard score >= *threshold*, sorted best-first.
    """
    scored = [
        (p, a, token_jaccard(hyp_pred, p))
        for p, a in ctx_preds.items()
        if token_jaccard(hyp_pred, p) >= threshold
    ]
    scored.sort(key=lambda x: -x[2])
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Strategy 5 — WordNet semantic entailment bridge
# ---------------------------------------------------------------------------

# Module-level cache: (general_token, specific_token, max_depth) → score
_WN_SCORE_CACHE: dict[tuple[str, str, int], float] = {}


def _wn_available() -> bool:
    """Return True if NLTK and the WordNet corpus are both importable."""
    try:
        from nltk.corpus import wordnet as wn  # noqa: F401

        wn.synsets("test")  # triggers corpus load; raises if data missing
        return True
    except Exception:
        return False


def _wn_synsets(word: str):
    """Return WordNet synsets for *word*, preferring verbs then all POS."""
    from nltk.corpus import wordnet as wn

    syns = wn.synsets(word, pos=wn.VERB)
    return syns if syns else wn.synsets(word)


def _wn_entailment_score(
    general_token: str,
    specific_token: str,
    max_depth: int = 4,
) -> float:
    """Score how strongly *specific_token* is a semantic hyponym / entailment
    of *general_token*, in [0, 1].

    Two complementary checks are performed:

    1. **Hypernym path** — does *specific_token*'s synset have *general_token*'s
       synset on one of its hypernym paths?  Depth from specific → general is
       mapped to a score that decays linearly to 0 at *max_depth*.

    2. **Verb entailment** — does any synset of *specific_token* directly
       entail any synset of *general_token*?  WordNet encodes relations like
       ``kill.v.01 → die.v.01`` explicitly.

    A high score means ``specific(X) → general(X)`` is semantically plausible,
    so the Prolog bridge ``general_pred :- specific_pred`` is logically sound.

    Returns 0.0 if WordNet is unavailable or the tokens are unrelated.
    """
    cache_key = (general_token, specific_token, max_depth)
    if cache_key in _WN_SCORE_CACHE:
        return _WN_SCORE_CACHE[cache_key]

    if not _wn_available():
        _WN_SCORE_CACHE[cache_key] = 0.0
        return 0.0

    if general_token == specific_token:
        _WN_SCORE_CACHE[cache_key] = 1.0
        return 1.0

    specific_syns = _wn_synsets(specific_token)
    general_syns = _wn_synsets(general_token)
    if not specific_syns or not general_syns:
        _WN_SCORE_CACHE[cache_key] = 0.0
        return 0.0

    best = 0.0

    for s_syn in specific_syns:
        # --- Check 1: hypernym path (troponym hierarchy) ---
        # hypernym_paths() returns lists ordered [root, ..., s_syn], so s_syn
        # is always the last element.  Depth from s_syn up to g_syn is:
        #   depth = (len(path) - 1) - path.index(g_syn)
        # Example: drown.v.03 path = [change, change_state, die, drown]
        #          depth to die = (4-1) - 2 = 1  (drown is direct hyponym of die)
        for path in s_syn.hypernym_paths():
            leaf_pos = len(path) - 1  # position of s_syn (always last)

            for g_syn in general_syns:
                if g_syn in path:
                    depth = leaf_pos - path.index(g_syn)
                    if depth <= max_depth:
                        score = 1.0 - depth / (max_depth + 1)
                        best = max(best, score)

            # --- Check 3 (transitive causation via hypernym path) ---
            # For each ancestor on the path, check whether that ancestor
            # *causes* any general synset.  This handles chains like:
            #   murder → kill (hypernym, 1 hop) → die (causes relation)
            for path_syn in path[:-1]:  # skip s_syn itself (handled directly)
                hops_up = leaf_pos - path.index(path_syn)
                if hops_up > max_depth:
                    continue
                for caused_syn in path_syn.causes():
                    if caused_syn in general_syns:
                        score = 0.9 * (1.0 - hops_up / (max_depth + 1))
                        best = max(best, score)

        # --- Check 2: direct verb entailment relation ---
        # WordNet encodes actions implied by performing the specific verb,
        # e.g. snore.v.01.entailments() → [sleep.v.01].
        for entailed_syn in s_syn.entailments():
            if entailed_syn in general_syns:
                best = max(best, 0.9)  # direct entailment: near-max score

        # --- Check 3 (direct): verb causation relation ---
        # WordNet encodes states/events caused by the specific verb, e.g.
        # kill.v.01.causes() → [die.v.01].  Canonical "kill entails die".
        for caused_syn in s_syn.causes():
            if caused_syn in general_syns:
                best = max(best, 0.9)  # direct causation: near-max score

    _WN_SCORE_CACHE[cache_key] = best
    return best


def wordnet_predicate_score(
    hyp_pred: str,
    ctx_pred: str,
    max_depth: int = 4,
    stop: frozenset[str] = _DEFAULT_STOP,
) -> float:
    """Score *ctx_pred* as a semantic hyponym of *hyp_pred*.

    For each content token in *hyp_pred*, find the highest-scoring matching
    token in *ctx_pred* using :func:`_wn_entailment_score`.  The final score
    is the mean of per-hyp-token best scores.

    A high score means ``ctx_pred(X,…) → hyp_pred(X,…)`` is semantically
    plausible, supporting the bridge rule ``hyp_pred :- ctx_pred``.

    Stop-words are stripped before comparison so that common prepositions /
    auxiliaries do not pollute the signal.
    """
    hyp_tokens = [t for t in hyp_pred.split("_") if t and t not in stop]
    ctx_tokens = [t for t in ctx_pred.split("_") if t and t not in stop]

    if not hyp_tokens or not ctx_tokens:
        return 0.0

    total = 0.0
    for ht in hyp_tokens:
        # Best score over all ctx tokens for this hyp token
        best = max(
            (_wn_entailment_score(ht, ct, max_depth) for ct in ctx_tokens),
            default=0.0,
        )
        total += best

    return total / len(hyp_tokens)


def find_wordnet_matches(
    hyp_preds: dict[str, int],
    ctx_preds: dict[str, int],
    threshold: float = 0.3,
    max_depth: int = 4,
    stop: frozenset[str] = _DEFAULT_STOP,
) -> list[tuple[str, str, int, int, float]]:
    """Return ``(hyp_pred, ctx_pred, hyp_arity, ctx_arity, score)`` tuples
    where *ctx_pred* is a semantic hyponym of *hyp_pred* with score >=
    *threshold*, sorted best-first per hypothesis predicate.

    Bridge direction: ``hyp_pred :- ctx_pred``
    (the more-specific context fact entails the more-general hypothesis goal).
    """
    results: list[tuple[str, str, int, int, float]] = []
    for hyp_pred, hyp_arity in hyp_preds.items():
        best_ctx, best_arity, best_score = None, 0, 0.0
        for ctx_pred, ctx_arity in ctx_preds.items():
            if hyp_pred == ctx_pred:
                continue
            score = wordnet_predicate_score(hyp_pred, ctx_pred, max_depth, stop)
            if score >= threshold and score > best_score:
                best_ctx, best_arity, best_score = ctx_pred, ctx_arity, score
        if best_ctx is not None:
            results.append((hyp_pred, best_ctx, hyp_arity, best_arity, best_score))
    results.sort(key=lambda x: -x[4])
    return results


# ---------------------------------------------------------------------------
# Bridge rule generation
# ---------------------------------------------------------------------------


def _anon_args(arity: int) -> str:
    """Return a comma-separated list of anonymous Prolog variables."""
    return ", ".join(f"_A{i}" for i in range(arity))


def make_bridge_rule(
    hyp_pred: str,
    hyp_arity: int,
    ctx_pred: str,
    ctx_arity: int,
) -> str:
    """Generate a forwarding bridge rule:
    ``hyp_pred(Args...) :- ctx_pred(Args...).``

    Uses anonymous variables for all arguments so the rule fires whenever
    *ctx_pred* is provable, regardless of argument values.  For variable
    goals (hypothesis contains Prolog variables) this enables provability
    checks; binding specific values requires a manual bridge rule.
    """
    hyp_args = _anon_args(hyp_arity)
    ctx_args = _anon_args(ctx_arity)
    head = f"{hyp_pred}({hyp_args})" if hyp_arity else hyp_pred
    body = f"{ctx_pred}({ctx_args})" if ctx_arity else ctx_pred
    return f"{head} :- {body}."


# ---------------------------------------------------------------------------
# Program injection
# ---------------------------------------------------------------------------


def inject_bridge_rules(program: str, rules: list[str], label: str = "") -> str:
    """Append *rules* to *program* under an optional comment label."""
    if not rules:
        return program
    header = f"\n% --- alignment bridge rules{' (' + label + ')' if label else ''} ---"
    clauses = "\n".join(r if r.strip().endswith(".") else r.strip() + "." for r in rules)
    return program + f"\n{header}\n{clauses}\n"


# ---------------------------------------------------------------------------
# Alignment config
# ---------------------------------------------------------------------------


@dataclass
class AlignmentConfig:
    """Configuration for the five-strategy alignment layer.

    Strategies are applied in order:
    ``normalize`` → ``aliases`` → ``bridge_rules`` → ``auto_bridge``
    → ``semantic_bridge``.
    Disable any strategy by leaving it at its default (falsy) value.

    Attributes:
        normalize:
            Strip English stop-words from predicate token lists and inject
            bridge rules where normalised forms match.
        stop_tokens:
            Custom stop-word set used by the normalisation and semantic-bridge
            strategies.  Defaults to a built-in set of ~50 common English
            words.
        aliases:
            Explicit ``{hyp_predicate_name: ctx_predicate_name}`` mapping.
            Each entry becomes a Prolog bridge rule connecting the two.
            Arities are inferred from the program; fall back to 0 if unknown.
        bridge_rules:
            Verbatim Prolog clauses injected as-is.  Use this for
            hand-crafted bridges that carry precise argument mappings, e.g.::

                "time_to_hear_back_from_court(_, within_10_days)"
                " :- court_sends_case_number_within_10_days(_, _)."

        auto_bridge:
            Automatically search for context predicates with high token
            overlap to each undefined hypothesis predicate and inject
            anonymous-variable bridge rules.
        auto_bridge_threshold:
            Minimum Jaccard token-overlap score to accept a candidate
            (default 0.2).
        auto_bridge_top_k:
            Maximum number of context candidates to bridge per hypothesis
            predicate (default 1 = best match only).
        semantic_bridge:
            Use WordNet hyponym / hypernym paths and verb-entailment relations
            to find context predicates that are semantically more specific than
            each undefined hypothesis predicate, then inject bridge rules.

            The bridge direction ``hyp :- ctx`` is logically sound when
            *ctx* tokens are hyponyms of *hyp* tokens (the more-specific
            context fact entails the more-general hypothesis goal).

            Requires ``nltk`` with the WordNet corpus::

                python -m nltk.downloader wordnet

        semantic_bridge_threshold:
            Minimum WordNet entailment score to accept a candidate
            (default 0.3).  Scores are in [0, 1]; 1.0 = same synset,
            values decay with hypernym path depth.
        semantic_bridge_max_depth:
            Maximum hypernym path depth to consider (default 4).  Deeper
            paths are treated as unrelated (score 0.0).
    """

    normalize: bool = False
    stop_tokens: frozenset[str] = field(default_factory=lambda: _DEFAULT_STOP)

    aliases: dict[str, str] = field(default_factory=dict)

    bridge_rules: list[str] = field(default_factory=list)

    auto_bridge: bool = False
    auto_bridge_threshold: float = 0.2
    auto_bridge_top_k: int = 1

    semantic_bridge: bool = False
    semantic_bridge_threshold: float = 0.3
    semantic_bridge_max_depth: int = 4

    constant_align: bool = False
    constant_align_types: tuple[str, ...] = ("person",)


# ---------------------------------------------------------------------------
# Strategy 6 — Typed constant alignment helpers
# ---------------------------------------------------------------------------


def _extract_typed_constants(
    kb: dict[str, Any],
    types: tuple[str, ...] = ("person",),
) -> set[str]:
    """Extract constant ids of specified types from a logic_kb dict."""
    return {
        c["id"]
        for c in kb.get("constants", [])
        if c.get("type") in types
    }


def _variabilize_constants_in_clause(
    clause: str,
    constants_to_replace: set[str],
) -> str:
    """Replace specified ground constants with _ in a ground fact's arguments.

    Only operates on ground facts (no :-). Rules with variables are left
    untouched since their variables already unify with anything.
    """
    if ":-" in clause:
        return clause
    m = re.match(r"(\s*[a-z][a-z0-9_]*\s*)\(([^)]+)\)(\s*\.?\s*)$", clause)
    if not m:
        return clause
    pred_part, args_str, tail = m.group(1), m.group(2), m.group(3)
    args = [a.strip() for a in args_str.split(",")]
    new_args = ["_" if a in constants_to_replace else a for a in args]
    return f"{pred_part}({', '.join(new_args)}){tail}"


def _variabilize_constants_in_goal(
    goal: str,
    constants_to_replace: set[str],
    restrict_to_preds: dict[str, int] | None = None,
) -> str:
    """Replace person constants in a (possibly compound) Prolog goal.

    If *restrict_to_preds* is given, only replace in atoms whose predicate
    name appears in that dict.  This prevents false positives for predicates
    that only exist in the scenario KB.
    """
    def _replace_in_atom(m: re.Match) -> str:
        pred = m.group(1)
        args_str = m.group(2)
        if restrict_to_preds is not None and pred not in restrict_to_preds:
            return m.group(0)
        args = [a.strip() for a in args_str.split(",")]
        new_args = ["_" if a in constants_to_replace else a for a in args]
        return f"{pred}({', '.join(new_args)})"
    return re.sub(r"([a-z][a-z0-9_]*)\(([^)]*)\)", _replace_in_atom, goal)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def align(
    program: str,
    goal: str,
    ctx_prolog: dict[str, Any] | None,
    sq_prolog: dict[str, Any],
    config: AlignmentConfig,
    ctx_kb: dict[str, Any] | None = None,
    sq_kb: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Apply all enabled alignment strategies and return ``(program, goal)``.

    The original KB text is never modified; all alignment is done by
    appending bridge rules to the end of *program*.

    Args:
        program:    Merged Prolog program string (from
                    :func:`~symbolic_conqa.prolog_checker.build_prolog_program`).
        goal:       Hypothesis goal string (with or without trailing dot).
        ctx_prolog: Context KB ``prolog`` block dict, or ``None``.
        sq_prolog:  Scenario KB ``prolog`` block dict.
        config:     Which strategies to apply and their parameters.

    Returns:
        ``(aligned_program, goal)`` — the goal string is returned unchanged
        (alignment only adds bridge rules to the program).
    """
    ctx_preds: dict[str, int] = (
        _extract_preds_from_block(ctx_prolog) if ctx_prolog else {}
    )
    sq_preds: dict[str, int] = _extract_preds_from_block(sq_prolog)
    defined_preds: dict[str, int] = _extract_defined_predicates(program)
    goal_preds: dict[str, int] = _extract_preds_from_goal(goal.rstrip("."))

    aligned = program

    # ------------------------------------------------------------------
    # Strategy 6 — Typed constant alignment (executed FIRST)
    # ------------------------------------------------------------------
    if config.constant_align:
        ctx_persons = _extract_typed_constants(
            ctx_kb or {}, config.constant_align_types
        )
        sq_persons = _extract_typed_constants(
            sq_kb or {}, config.constant_align_types
        )

        if ctx_persons:
            # Variabilize person args in context ground facts
            lines = aligned.split("\n")
            new_lines = []
            in_context = False
            for line in lines:
                if "% --- context KB ---" in line:
                    in_context = True
                elif "% --- scenario KB ---" in line:
                    in_context = False
                if in_context and line.strip() and not line.strip().startswith("%"):
                    line = _variabilize_constants_in_clause(line, ctx_persons)
                new_lines.append(line)
            aligned = "\n".join(new_lines)

        if sq_persons:
            # Variabilize person constants in the goal, but ONLY for
            # predicates that also appear in the context KB.  This
            # prevents false positives from scenario-only predicates
            # where the person arg slot might match a non-person arg
            # in an unrelated context fact (e.g. pred(me) matching
            # pred(true)).
            goal = _variabilize_constants_in_goal(
                goal, sq_persons, restrict_to_preds=ctx_preds,
            )

    # ------------------------------------------------------------------
    # Strategy 1 — Normalisation
    # ------------------------------------------------------------------
    if config.normalize and ctx_preds:
        matches = find_normalized_matches(goal_preds, ctx_preds, config.stop_tokens)
        rules = [
            make_bridge_rule(hp, ha, cp, ca)
            for hp, cp, ha, ca in matches
        ]
        if rules:
            aligned = inject_bridge_rules(aligned, rules, "normalize")

    # ------------------------------------------------------------------
    # Strategy 2 — Explicit aliases
    # ------------------------------------------------------------------
    if config.aliases:
        rules = []
        for hyp_pred, ctx_pred in config.aliases.items():
            hyp_arity = goal_preds.get(hyp_pred, defined_preds.get(hyp_pred, 0))
            ctx_arity = defined_preds.get(ctx_pred, ctx_preds.get(ctx_pred, 0))
            rules.append(make_bridge_rule(hyp_pred, hyp_arity, ctx_pred, ctx_arity))
        aligned = inject_bridge_rules(aligned, rules, "aliases")

    # ------------------------------------------------------------------
    # Strategy 3 — Verbatim bridge rules
    # ------------------------------------------------------------------
    if config.bridge_rules:
        aligned = inject_bridge_rules(aligned, config.bridge_rules, "manual")

    # ------------------------------------------------------------------
    # Strategy 4 — Auto-bridge via token overlap
    # ------------------------------------------------------------------
    if config.auto_bridge and ctx_preds:
        # Only attempt auto-bridge for predicates called in the goal that
        # are not already defined anywhere in the program.
        undefined_hyp = {
            p: a for p, a in goal_preds.items() if p not in defined_preds
        }
        rules = []
        for hyp_pred, hyp_arity in undefined_hyp.items():
            candidates = find_best_ctx_match(
                hyp_pred,
                ctx_preds,
                threshold=config.auto_bridge_threshold,
                top_k=config.auto_bridge_top_k,
            )
            for ctx_pred, ctx_arity, score in candidates:
                rules.append(make_bridge_rule(hyp_pred, hyp_arity, ctx_pred, ctx_arity))
        if rules:
            aligned = inject_bridge_rules(aligned, rules, "auto_bridge")

    # ------------------------------------------------------------------
    # Strategy 5 — WordNet semantic entailment bridge
    # ------------------------------------------------------------------
    if config.semantic_bridge and ctx_preds:
        # Refresh defined_preds to include any bridge rules added above
        defined_preds_now = _extract_defined_predicates(aligned)
        undefined_hyp = {
            p: a for p, a in goal_preds.items() if p not in defined_preds_now
        }
        if undefined_hyp:
            matches = find_wordnet_matches(
                undefined_hyp,
                ctx_preds,
                threshold=config.semantic_bridge_threshold,
                max_depth=config.semantic_bridge_max_depth,
                stop=config.stop_tokens,
            )
            rules = [
                make_bridge_rule(hp, ha, cp, ca)
                for hp, cp, ha, ca, _score in matches
            ]
            if rules:
                aligned = inject_bridge_rules(aligned, rules, "semantic_bridge")

    return aligned, goal
