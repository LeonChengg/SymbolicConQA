# Symbolic ConQA

Extract formal logic from text using LLMs, then use SWI-Prolog entailment to answer conditional questions from the [ConditionalQA](https://haitian99.github.io/conditionalqa/) dataset.

**Pipeline overview:**

```
documents.json
    │
    ├─ (optional) data_propocessing.py ──► documents_with_semantic_tree.json
    │
    └─ extract.py (context) ─────────────► all_context_with_logic.jsonl
                                                    │
dev.json                                            │
    └─ extract.py (scenario_question) ──► all_SQ_with_logic_yesno.jsonl
                                                    │
                                    ┌───────────────┴────────────────┐
                                    │                                │
                            answer_questions.py            llm_answer_questions.py
                            (Prolog entailment)            (LLM baseline)
                                    │                                │
                            all_answers.jsonl              llm_answers.jsonl
                                    │                                │
                                    └───────────┬────────────────────┘
                                        evaluate_answers.py
                                        (EM / F1 / P / R)
```

## Project Structure

```
SymbolicConQA/
├── src/symbolic_conqa/
│   ├── extraction.py        # Text extractors + LLM batch extraction pipeline
│   ├── alignment.py         # 6-strategy alignment layer for Prolog bridging
│   ├── prolog_checker.py    # SWI-Prolog entailment engine
│   ├── models.py            # Pydantic models (LogicKB, LogicKBList)
│   ├── prompts.py           # LLM prompt templates
│   └── io_utils.py          # load_json_or_jsonl / write_jsonl helpers
├── scripts/
│   ├── extract.py               # CLI: extract FOL/Prolog from documents or SQ
│   ├── answer_questions.py      # CLI: answer questions via Prolog entailment
│   ├── llm_answer_questions.py  # CLI: LLM baseline (HuggingFace models)
│   ├── evaluate_answers.py      # CLI: compute EM/F1 vs ConditionalQA gold
│   ├── check_entailment.py      # CLI: run Prolog entailment over all SQ
│   └── test_entailment.py       # Smoke-tests (13/13) for the entailment checker
├── data/ConditionalQA/v1_0/
│   ├── documents.json                        # 652 UK gov policy documents (HTML)
│   ├── documents_with_semantic_tree.json     # same + semantic_tree key (generated)
│   ├── dev.json                              # ConditionalQA dev split
│   ├── train.json
│   ├── documents_sq_subset.json
│   ├── evaluate.py                           # Official ConditionalQA evaluator
│   └── data_propocessing.py                  # HTML → semantic tree converter
├── results/                     # Output files (large files git-ignored)
│   ├── all_context_with_logic.jsonl
│   ├── all_SQ_with_logic_yesno.jsonl
│   ├── all_answers.jsonl
│   └── llm_answers.jsonl
├── pyproject.toml
└── uv.lock
```

## Quick Start

### 1. Install Dependencies

This project uses [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

SWI-Prolog is required for entailment checking:

```bash
# Debian/Ubuntu
sudo apt install swi-prolog

# or via conda
conda install -c conda-forge swi-prolog
```

WordNet is required for the semantic alignment strategy:

```bash
uv run python -c "import nltk; nltk.download('wordnet')"
```

### 2. Configure Environment Variables

```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

---

## Step 1 — Data Preprocessing (optional)

Convert the raw HTML `contents` field in `documents.json` into a structured **semantic tree** for cleaner LLM input.

```bash
# Generate documents_with_semantic_tree.json (652 documents)
uv run python data/ConditionalQA/v1_0/data_propocessing.py

# Custom paths
uv run python data/ConditionalQA/v1_0/data_propocessing.py \
    -i data/ConditionalQA/v1_0/documents.json \
    -o data/ConditionalQA/v1_0/documents_with_semantic_tree.json

# Demo: print the outline and semantic tree for a built-in example
uv run python data/ConditionalQA/v1_0/data_propocessing.py --demo
```

The semantic tree converts HTML tags into typed nodes:

| Node type | Content |
|---|---|
| `section` | Heading text + nested children |
| `text` | Plain paragraph |
| `rule` | Eligibility rule with `outcome` (ELIGIBLE / NOT_ELIGIBLE) + condition list |
| `list` | Bullet items with optional `operator` (ANY / ALL) and `exceptions` |
| `condition` | A single condition extracted from a list |
| `table` | Column headers + row objects (handles bare `<tr>cell\|cell</tr>` format) |

---

## Step 2 — Logic Extraction

Extract FOL and Prolog knowledge bases from text using an LLM.

### Context documents

```bash
# From raw HTML (default)
uv run python scripts/extract.py context

# From semantic tree (cleaner input — requires Step 1)
uv run python scripts/extract.py context --input-field semantic_tree

# Custom paths, model, and batch size
uv run python scripts/extract.py context \
    --input-field semantic_tree \
    -i data/ConditionalQA/v1_0/documents_with_semantic_tree.json \
    -o results/all_context_with_logic.jsonl \
    -m gpt-4o -b 10
```

### Scenario-question pairs

```bash
# All questions
uv run python scripts/extract.py scenario_question

# Yes/no questions only
uv run python scripts/extract.py scenario_question --yes-no-only \
    -o results/all_SQ_with_logic_yesno.jsonl

# Inject context KB predicates/constants for better alignment
uv run python scripts/extract.py scenario_question --yes-no-only \
    --context-kb results/all_context_with_logic.jsonl
```

### `extract.py` options

| Flag | Short | Description | Default |
|---|---|---|---|
| `--input` | `-i` | Input file path | task-specific |
| `--output` | `-o` | Output file path | task-specific |
| `--model` | `-m` | OpenAI model | `gpt-5-mini` |
| `--batch-size` | `-b` | Items per LLM batch | `5` |
| `--num-batches` | `-n` | Batches to process (all if omitted) | — |
| `--input-field` | | `contents` or `semantic_tree` | `contents` |
| `--yes-no-only` | | Filter to yes/no questions only | off |
| `--context-kb` | | Context KB JSONL to inject predicates from | — |

Extraction is **crash-resumable**: re-running picks up from where it left off.

### Output record format

```json
{
  "index": 0,
  "id": "dev-0",
  "logic_kb": {
    "constants": [{"id": "me", "type": "person", "gloss": "the applicant"}],
    "predicates": [{"name": "eligible_for_benefit", "arity": 1, "gloss": "..."}],
    "fol":    {"facts": [...], "rules": [...], "hypothesis": "..."},
    "prolog": {"facts": [...], "rules": [...], "hypothesis": "..."}
  },
  "data": {"url": "...", "question": "...", "answers": [...]}
}
```

---

## Step 3a — Answer Questions via Prolog Entailment

```bash
# Basic run (no alignment)
uv run python scripts/answer_questions.py \
    -c results/all_context_with_logic.jsonl \
    -s results/all_SQ_with_logic_yesno.jsonl \
    -o results/all_answers.jsonl

# With all alignment strategies enabled
uv run python scripts/answer_questions.py \
    --normalize \
    --auto-bridge --auto-bridge-threshold 0.25 \
    --semantic-bridge --semantic-bridge-threshold 0.3 --semantic-bridge-pos verb_noun \
    --constant-align
```

### Alignment strategies

Predicate name mismatches between the hypothesis and context KB are bridged by injecting Prolog rules automatically. Six strategies are applied in order:

| Strategy | Flag | Description |
|---|---|---|
| 1. Normalize | `--normalize` | Strip stop-words from predicate tokens; bridge matching normalized forms |
| 2. Aliases | `--alias hyp=ctx` | Explicit predicate mapping (repeatable) |
| 3. Bridge rules | `--bridge-rule "h(X):-c(X)."` | Verbatim Prolog clauses (repeatable) |
| 4. Auto-bridge | `--auto-bridge` | Fuzzy token-overlap (Jaccard) matching |
| 5. Semantic bridge | `--semantic-bridge` | WordNet hypernym/hyponym + verb entailment/causation |
| 6. Constant align | `--constant-align` | Variabilize person-typed constants for alignment |

### WordNet POS mode (`--semantic-bridge-pos`)

| Value | Synsets used | Best for |
|---|---|---|
| `verb_noun` | verb + noun (default) | Mixed action + entity predicates |
| `verb` | verb only | Action-only predicate names |
| `noun` | noun only | Entity/category predicate names |

### `answer_questions.py` options

| Flag | Description | Default |
|---|---|---|
| `-c / --context` | Context KB JSONL | `results/all_context_with_logic.jsonl` |
| `-s / --sq` | Question JSONL | `results/all_SQ_with_logic_yesno.jsonl` |
| `-o / --output` | Output JSONL | `results/all_answers.jsonl` |
| `-t / --timeout` | Per-query Prolog timeout (seconds) | `10` |
| `--normalize` | Enable stop-word normalisation | off |
| `--auto-bridge` | Enable fuzzy token-overlap bridging | off |
| `--auto-bridge-threshold` | Jaccard threshold | `0.2` |
| `--alias hyp=ctx` | Explicit predicate alias (repeatable) | — |
| `--bridge-rule "…"` | Verbatim Prolog bridge clause (repeatable) | — |
| `--semantic-bridge` | Enable WordNet semantic bridge | off |
| `--semantic-bridge-threshold` | Min entailment score | `0.3` |
| `--semantic-bridge-max-depth` | Max hypernym path depth | `4` |
| `--semantic-bridge-pos` | POS mode: `verb`, `noun`, `verb_noun` | `verb_noun` |
| `--constant-align` | Variabilize person-typed constants | off |

---

## Step 3b — LLM Baseline

Answer questions using a local HuggingFace model (no Prolog required):

```bash
uv run python scripts/llm_answer_questions.py \
    -m /path/to/Llama-3.1-8B-Instruct \
    -s results/all_SQ_with_logic_yesno.jsonl \
    -c results/all_context_with_logic.jsonl \
    -o results/llm_answers.jsonl \
    --few-shot
```

### `llm_answer_questions.py` options

| Flag | Description | Default |
|---|---|---|
| `-m / --model` | Path to local HuggingFace model | Llama-3.1-8B-Instruct |
| `-s / --sq` | Question JSONL | `results/all_SQ_with_logic_yesno.jsonl` |
| `-c / --context` | Context JSONL | `results/all_context_with_logic.jsonl` |
| `-o / --output` | Output JSONL | `results/llm_answers.jsonl` |
| `--max-new-tokens` | Max tokens to generate | `16` |
| `--max-context-chars` | Max context characters in prompt | `3000` |
| `--device` | Device map (`auto`, `cpu`, `cuda`) | `auto` |
| `--batch-size` | Inference batch size | `1` |
| `--few-shot` | Prepend 3 labelled yes/no/not_answerable examples | off |

The model receives: system prompt → (optional few-shot examples) → context + scenario + question, and must reply with exactly one word: `yes`, `no`, or `not_answerable`.

---

## Step 4 — Evaluate

```bash
# Evaluate Prolog results (auto-detected)
uv run python scripts/evaluate_answers.py -p results/all_answers.jsonl

# Evaluate LLM results (auto-detected from 'raw_output' key)
uv run python scripts/evaluate_answers.py -p results/llm_answers.jsonl

# Force mode explicitly
uv run python scripts/evaluate_answers.py -p results/llm_answers.jsonl --mode llm

# Save per-question breakdown to JSON
uv run python scripts/evaluate_answers.py -p results/all_answers.jsonl \
    --per-question results/eval_breakdown.json
```

The evaluator reports:
- **Yes/No accuracy** — exact match against gold yes/no answers
- **EM / F1** — exact match and token F1 for value questions (via official ConditionalQA script)
- **EM+conditions / F1+conditions** — evaluation with conditional answer conditions
- **Precision / Recall** — across yes/no and not-answerable question types
- **Prediction distribution** — breakdown of yes / no / not-answerable predictions

---

## Development

```bash
# Format
uv run ruff format src/ scripts/

# Lint
uv run ruff check src/ scripts/

# Type check
uv run basedpyright src/ scripts/

# Run entailment smoke-tests (13 tests)
uv run python scripts/test_entailment.py
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `openai` | LLM API for logic extraction |
| `pydantic` | Structured output parsing (LogicKB schema) |
| `typer` + `rich` | CLI and progress display |
| `transformers` + `torch` | LLM baseline inference |
| `nltk` (WordNet) | Semantic bridge alignment strategy |
| `beautifulsoup4` + `lxml` | HTML parsing for semantic tree preprocessing |
| `python-dotenv` | `.env` file loading |
