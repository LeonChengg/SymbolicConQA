"""Prompt templates for logic extraction."""

_SYSTEM_PROMPT_BASE = """You are an information extraction engine that converts short English narratives into formal logic.

TASK
Given an input text, extract:
1) A First-Order Logic (FOL) representation (facts + optional rules).
2) A Prolog representation (facts + optional rules).
3) A symbol table (constants and predicates with arity and gloss).
{hypothesis_line}
REQUIREMENTS
- Be faithful to the text. Do NOT add commonsense/legal inferences unless explicitly stated.
- Represent uncertainty/hedges explicitly:
  - "apparently X" => apparently(X) in Prolog and Apparently(X) in FOL (or a consistent variant).
  - "I believe X" => believes(me, X) in Prolog and Believes(me, X) in FOL.
- Resolve obvious coreference for "my", "I", "my father", "my uncle".
- Use short lowercase constants: me, f, u, wales, t_recent...
- Predicates must be lowercase_with_underscores in Prolog; FOL can use CamelCase or keep consistent.
- Output MUST conform exactly to the provided JSON schema; do not include any extra keys.
- Include optional_rules only if you are explicitly labeling them as optional (e.g., kinship expansion).
- In Prolog facts and rules, when an argument represents a GENERIC role (any person,
  any applicant, any worker, anyone) rather than a specific named individual, use an
  UPPERCASE Prolog variable (e.g., Person, X, Applicant) instead of a lowercase ground
  atom. This ensures the fact can unify with any specific entity at query time.
  - CORRECT: must_be_aged_18_or_over(Person).     % applies to anyone
  - CORRECT: can_write_request_to(Person, cac).    % any person can write to CAC
  - WRONG:   must_be_aged_18_or_over(applicant).   % ground atom, won't match 'me'
  - WRONG:   can_write_request_to(person, cac).    % 'person' is a ground atom in Prolog
  - Keep specific named entities as lowercase constants: queen, hmrc, uk, etc.

CRITICAL FOR CONTEXT DOCUMENTS (general policy / legal texts):
Context documents describe general policies applying to anyone.
NEVER use "me", "you", or any ground atom for the person role in rules or general facts.
ALWAYS use uppercase Prolog variables (Person, P, Applicant, etc.) for the person slot.
Only use ground person atoms (me, john, etc.) in scenario-specific facts about a particular individual."""

SYSTEM_PROMPT = _SYSTEM_PROMPT_BASE.format(
    hypothesis_line=(
        "4) A hypothesis in both FOL and Prolog forms (placed in the `hypothesis` field"
        " inside the `fol` and `prolog` blocks respectively).\n"
    ),
) + """

EXAMPLE (scenario/question extraction with context predicates):
Context predicates: can_apply_for_blue_badge/1, aged_3_or_over/1, has_permanent_disability/1, affects_mobility/1
Context rules: can_apply_for_blue_badge(Person) :- aged_3_or_over(Person), has_permanent_disability(Person), affects_mobility(Person).
Input scenario: "My grandmother is 72 years old and has arthritis that makes it hard for her to walk."
Input question: "Can my grandmother apply for a Blue Badge?"
Output (Prolog):
  facts:
    - "aged_3_or_over(grandmother)."
    - "has_permanent_disability(grandmother)."
    - "affects_mobility(grandmother)."
  rules: []
  hypothesis: "can_apply_for_blue_badge(grandmother)"
  Note: Facts use the EXACT predicate names from context. Hypothesis is the HEAD of a matching context rule.
  Note: "grandmother" is a specific entity from the scenario, so it is a lowercase constant.
  Note: Facts are extracted because the scenario TEXT supports them (72 > 3, arthritis = disability, hard to walk = affects mobility).
"""

SYSTEM_PROMPT_NO_HYPOTHESIS = _SYSTEM_PROMPT_BASE.format(hypothesis_line="") + """

EXAMPLE (context document extraction):
Input: "You can apply for a Blue Badge if you are aged 3 or over and have a permanent disability that affects your mobility."
Output (Prolog):
  facts: []
  rules:
    - "can_apply_for_blue_badge(Person) :- aged_3_or_over(Person), has_permanent_disability(Person), affects_mobility(Person)."
  Note: Person is an uppercase Prolog variable — it unifies with ANY specific entity (me, john, etc.).
  Note: No facts here because the text states general eligibility criteria, not specific individuals.
Output (FOL):
  facts: []
  rules:
    - "∀x (Aged3OrOver(x) ∧ HasPermanentDisability(x) ∧ AffectsMobility(x) → CanApplyForBlueBadge(x))"
"""

USER_TEMPLATE = """Extract logic from the following text.

TEXT:
{input_text}

CONSTRAINTS:
- Return only data conforming to the schema.
- Do not infer anything not stated, except optional_rules (clearly separated).
- Keep it minimal and consistent.
"""

USER_TEMPLATE_WITH_CONTEXT_PREDICATES = """Extract logic from the following text.

TEXT:
{input_text}

CONTEXT PREDICATES (from the corresponding context document's KB):
{context_predicates}

CONTEXT CONSTANTS (from the context document's KB):
{context_constants}

CONTEXT RULES (Prolog rules from the context document):
{context_rules}

CONSTRAINTS:
- Return only data conforming to the schema.
- Do not infer anything not stated, except optional_rules (clearly separated).
- Keep it minimal and consistent.
- IMPORTANT: When the scenario/question refers to concepts that match the context predicates above,
  you MUST reuse the EXACT predicate name and arity from the context. Do NOT invent new predicate
  names for concepts already covered by the context predicates.
- You may introduce NEW predicates only for concepts NOT covered by any context predicate.
- The hypothesis MUST use context predicate names where applicable.
- IMPORTANT: When the scenario/question refers to an entity that corresponds to one of the
  CONTEXT CONSTANTS above, you MUST use the EXACT constant id from the context. For example:
    - If the context uses "vaccine_damage_payment", use "vaccine_damage_payment" (not "vaccine_injury_payment").
    - If the context uses "you" for the applicant, use "you" (not "me" or "applicant").
    - If the context uses "hm_land_registry", use "hm_land_registry" (not "rural_land_register").
- For entities NOT covered by any context constant (e.g. a specific family member or object
  mentioned only in the scenario), you may introduce new constants.
- IMPORTANT: When extracting facts from the scenario, check whether the scenario text
  semantically implies any predicate used in the CONTEXT RULES' body. If so, use that
  exact predicate. For example, if a context rule requires "severely_disabled(P)" and
  the scenario describes someone with serious paralysis, extract "severely_disabled(me)".
- Do NOT assert facts just because a rule needs them — only if the scenario text supports them.
- The hypothesis should correspond to the HEAD of a relevant context rule where applicable.
- Do NOT include "?-" prefix or trailing "%% comments" in the Prolog hypothesis — output a clean Prolog term only.
"""

BATCH_USER_TEMPLATE = """You will process multiple items.

For each item i (delimited by "=== ITEM [i] ==="), apply the following USER_TEMPLATE exactly, with its {{input_text}} filled in:
---
{single_user_template}
---

Return ONLY JSON matching this schema exactly:
{{ "items": [ LogicKB, LogicKB, ... ] }}

ITEMS:
{indexed_texts}
"""
