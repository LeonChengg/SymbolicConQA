from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union, Optional, Literal
import os

from pydantic import BaseModel, Field
from openai import OpenAI


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# =========================
# 1) Your prompts (unchanged)
# =========================

SYSTEM_PROMPT = """You are an information extraction engine that converts short English narratives into formal logic.

TASK
Given an input text, extract:
1) A First-Order Logic (FOL) representation (facts + optional rules).
2) A Prolog representation (facts + optional rules).
3) A symbol table (constants and predicates with arity and gloss).
4) A hypothesize in FOL with previous generated predicates and FOL operators.

REQUIREMENTS
- Be faithful to the text. Do NOT add commonsense/legal inferences unless explicitly stated.
- Represent uncertainty/hedges explicitly:
  - "apparently X" => apparently(X) in Prolog and Apparently(X) in FOL (or a consistent variant).
  - "I believe X" => believes(me, X) in Prolog and Believes(me, X) in FOL.
- Resolve obvious coreference for "my", "I", "my father", "my uncle".
- Use short lowercase constants: me, f, u, wales, t_recent...
- Predicates must be lowercase_with_underscores in Prolog; FOL can use CamelCase or keep consistent.
- Output MUST conform exactly to the provided JSON schema; do not include any extra keys.
- Include optional_rules only if you are explicitly labeling them as optional (e.g., kinship expansion)."""

USER_TEMPLATE = """Extract logic from the following text.

TEXT:
{input_text}

CONSTRAINTS:
- Return only data conforming to the schema.
- Do not infer anything not stated, except optional_rules (clearly separated).
- Keep it minimal and consistent.
"""


# =========================
# 2) Output schema (must match what you expect)
# =========================

class Constant(BaseModel):
    id: str
    type: Literal["person", "location", "time", "organization", "object", "unknown"] = "unknown"
    gloss: str

class Predicate(BaseModel):
    name: str
    arity: int = Field(..., ge=1, le=8)
    gloss: str

class FOLBlock(BaseModel):
    facts: List[str] = Field(default_factory=list)
    rules: List[str] = Field(default_factory=list)
    optional_rules: List[str] = Field(default_factory=list)

class PrologBlock(BaseModel):
    facts: List[str] = Field(default_factory=list)
    rules: List[str] = Field(default_factory=list)
    optional_rules: List[str] = Field(default_factory=list)

class LogicKB(BaseModel):
    constants: List[Constant] = Field(default_factory=list)
    predicates: List[Predicate] = Field(default_factory=list)
    fol: FOLBlock
    prolog: PrologBlock
    hypothesis_fol: str = Field(..., description="FOL hypothesis for the question/claim")

class LogicKBList(BaseModel):
    items: List[LogicKB] = Field(default_factory=list)


# =========================
# 3) Batch wrapper (applies your USER_TEMPLATE per item)
# =========================

BATCH_USER_TEMPLATE = """You will process multiple items.

For each item i, apply the following USER_TEMPLATE exactly, with its {{input_text}} filled in:
---
{single_user_template}
---

Return ONLY JSON matching this schema exactly:
{{ "items": [ LogicKB, LogicKB, ... ] }}

ITEMS (indexed):
{indexed_texts}
"""



# =========================
# 4) IO helpers
# =========================

def load_json_or_jsonl(path: Union[str, Path]) -> List[Any]:
    path = Path(path)
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        return []

    # JSONL heuristic
    if "\n" in txt and not txt.lstrip().startswith("["):
        return [json.loads(line) for line in txt.splitlines() if line.strip()]

    data = json.loads(txt)
    if not isinstance(data, list):
        raise ValueError("Input file must be a JSON array or JSONL with one item per line.")
    return data


def chunked(lst: List[Any], size: int) -> List[List[Any]]:
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def build_input_text(sample: Dict[str, Any]) -> str:
    scenario = sample.get("scenario")
    question = sample.get("question")

    if not isinstance(scenario, str) or not scenario.strip():
        raise ValueError(f"Missing/empty 'scenario' in sample id={sample.get('id')}: {sample!r}")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Missing/empty 'question' in sample id={sample.get('id')}: {sample!r}")

    # as requested: scenario + question
    return scenario.strip() + "\n\n" + question.strip()


# =========================
# 5) Model call (true batching)
# =========================

def extract_logic_batch(client: OpenAI, texts: List[str], model: str) -> List[LogicKB]:
    indexed_texts = "\n".join([f"[{i}] {t}" for i, t in enumerate(texts)])

    # 1) Escape braces in USER_TEMPLATE so `{input_text}` doesn't get consumed by Python .format
    safe_user_template = USER_TEMPLATE.replace("{", "{{").replace("}", "}}").strip()

    # 2) Construct user message
    user_msg = BATCH_USER_TEMPLATE.format(
        single_user_template=safe_user_template,
        indexed_texts=indexed_texts,
    )

    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        text_format=LogicKBList,
    )
    return resp.output_parsed.items


# =========================
# 6) Runner: first 3 batches test, save originals under "data"
# =========================

def run_first_n_batches(
    in_path: str = "dev.json",
    out_path: str = "dev_with_logic.jsonl",
    model: str = "gpt-4.1",
    batch_size: int = 5,
    num_test_batches: int = 3,
) -> None:
    client = OpenAI()
    samples_any = load_json_or_jsonl(in_path)

    samples: List[Dict[str, Any]] = []
    for i, s in enumerate(samples_any):
        if not isinstance(s, dict):
            raise ValueError(f"Each sample must be a dict/object. Found {type(s)} at index {i}.")
        samples.append(s)

    if not samples:
        print("No samples found.")
        return

    # jobs: we keep original under 'data' in output (not 'input')
    jobs: List[Dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        jobs.append({
            "index": idx,
            "id": sample.get("id"),
            "input_text": build_input_text(sample),  # internal only, not saved
            "data": sample,                          # saved as "data"
        })

    batches = chunked(jobs, batch_size)
    total_batches = min(num_test_batches, len(batches))

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_file.open("w", encoding="utf-8") as f:
        for b in range(total_batches):
            batch_jobs = batches[b]
            batch_texts = [j["input_text"] for j in batch_jobs]

            kb_list = extract_logic_batch(client, batch_texts, model=model)
            if len(kb_list) != len(batch_jobs):
                raise RuntimeError(
                    f"Model returned {len(kb_list)} items but expected {len(batch_jobs)} in batch {b}."
                )

            for job, kb in zip(batch_jobs, kb_list):
                record = {
                    "index": job["index"],
                    "id": job["id"],
                    "logic_kb": kb.model_dump(),
                    "data": job["data"],   # original sample here
                }
                f.write(json.dumps(record, ensure_ascii=False, indent=4) + "\n\n")
                written += 1

            print(f"Finished batch {b+1}/{total_batches} ({len(batch_jobs)} items).")

    print(f"Saved {written} items to {out_path} (first {total_batches} batches only).")


if __name__ == "__main__":
    run_first_n_batches(
        in_path="data/ConditionalQA/v1_0/dev.json",
        out_path="dev_with_logic.jsonl",
        model="gpt-5-mini",
        batch_size=5,
        num_test_batches=1,
    )