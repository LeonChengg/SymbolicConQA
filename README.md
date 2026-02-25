# Symbolic ConQA

Extract formal logic from text using LLMs - converts English narratives into First-Order Logic (FOL) and Prolog representations.

## Project Structure

```
SymbolicConQA/
├── src/symbolic_conqa/          # Shared library code
│   ├── __init__.py
│   ├── models.py                # Pydantic model definitions
│   ├── prompts.py               # Prompt templates
│   ├── extraction.py            # Core extraction logic + text extractors
│   ├── io_utils.py              # IO utility functions (load/write JSONL)
│   └── prolog_checker.py        # SWI-Prolog entailment checker
├── scripts/
│   ├── extract.py               # CLI: extract FOL/Prolog from context & SQ
│   ├── check_entailment.py      # CLI: run Prolog entailment over all SQ records
│   └── test_entailment.py       # Smoke-tests for the entailment checker
├── data/
│   └── ConditionalQA/
│       ├── evaluate.py
│       └── v1_0/
│           ├── dev.json
│           ├── train.json
│           ├── documents.json
│           └── documents_sq_subset.json
├── results/                     # Output JSONL files (git-ignored large files)
│   ├── context_with_logic.jsonl
│   ├── SQ_with_logic.jsonl
│   └── entailment_results.jsonl
├── pyproject.toml               # Project configuration & dependencies
├── pyrightconfig.json           # Type-checker configuration
└── uv.lock                      # Locked dependency versions
```

## Quick Start

### 1. Install Dependencies

This project uses [uv](https://github.com/astral-sh/uv) for dependency management:

```bash
uv sync
```

### 2. Configure Environment Variables

Create a `.env` file and add your OpenAI API key:

```bash
OPENAI_API_KEY=your_api_key_here
```

## Usage

The unified CLI `scripts/extract.py` supports two tasks:

```bash
# Extract logic from context documents
uv run python scripts/extract.py context

# Extract logic from scenario-question pairs
uv run python scripts/extract.py scenario_question
```

### Options

| Flag | Short | Description | Default |
|---|---|---|---|
| `--input` | `-i` | Input file path | task-specific |
| `--output` | `-o` | Output file path | `results/context_with_logic.jsonl` or `results/SQ_with_logic.jsonl` |
| `--model` | `-m` | OpenAI model | `gpt-5-mini` |
| `--batch-size` | `-b` | Items per batch | `5` |
| `--num-batches` | `-n` | Number of batches (all if omitted) | `None` |

### Examples

```bash
# Run a quick test with 1 batch
uv run python scripts/extract.py context -n 1

# Custom output path and batch size
uv run python scripts/extract.py scenario_question -o results/custom.jsonl -b 10

# Use a different model
uv run python scripts/extract.py context -m gpt-4o
```

## Development

### Code Formatting

```bash
uv run ruff format src/ scripts/
```

### Code Linting

```bash
uv run ruff check src/ scripts/
```

### Type Checking

```bash
uv run basedpyright src/ scripts/
```

## Output Format

Results are written to `results/` as JSONL files. Each record contains:

```json
{
  "index": 0,
  "id": "sample_id",
  "logic_kb": {
    "constants": [...],
    "predicates": [...],
    "fol": {
      "facts": [...],
      "rules": [...],
      "optional_rules": [...],
      "hypothesis": "..."
    },
    "prolog": {
      "facts": [...],
      "rules": [...],
      "optional_rules": [...],
      "hypothesis": "..."
    }
  },
  "data": {...}
}
```
