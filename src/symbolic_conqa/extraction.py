"""Core logic extraction functionality."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from openai import OpenAI
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn

from .io_utils import chunked, load_json_or_jsonl, load_valid_records
from .models import LogicKB, LogicKBList
from .prompts import BATCH_USER_TEMPLATE, SYSTEM_PROMPT, SYSTEM_PROMPT_NO_HYPOTHESIS, USER_TEMPLATE

console = Console()

# ---------------------------------------------------------------------------
# Text extractors
# ---------------------------------------------------------------------------


def extract_text_from_contents(sample: dict[str, Any]) -> str:
    """Extract input text from a sample's ``contents`` field (str or list[str])."""
    contents = sample.get("contents")
    if isinstance(contents, list):
        text = "\n".join(str(item) for item in contents)
    elif isinstance(contents, str):
        text = contents
    else:
        raise ValueError(f"Missing/invalid 'contents' in sample id={sample.get('id')}")
    if not text.strip():
        raise ValueError(f"Empty 'contents' in sample id={sample.get('id')}")
    return text.strip()


def extract_text_from_scenario_question(sample: dict[str, Any]) -> str:
    """Extract input text by combining ``scenario`` and ``question`` fields."""
    scenario = sample.get("scenario")
    question = sample.get("question")
    if not isinstance(scenario, str) or not scenario.strip():
        raise ValueError(f"Missing/empty 'scenario' in sample id={sample.get('id')}: {sample!r}")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Missing/empty 'question' in sample id={sample.get('id')}: {sample!r}")
    return scenario.strip() + "\n\n" + question.strip()


# ---------------------------------------------------------------------------
# Shared extraction pipeline
# ---------------------------------------------------------------------------


def run_extraction(
    text_extractor: Callable[[dict[str, Any]], str],
    in_path: str,
    out_path: str,
    model: str = "gpt-5-mini",
    batch_size: int = 5,
    num_test_batches: int | None = None,
    *,
    include_hypothesis: bool = True,
    sample_filter: Callable[[dict[str, Any]], bool] | None = None,
) -> None:
    """
    Run logic extraction on a dataset with crash-resume support.

    Args:
        text_extractor: Function that pulls the input text from a sample dict.
        in_path: Input file path.
        out_path: Output file path.
        model: OpenAI model to use.
        batch_size: Number of items per batch.
        num_test_batches: Number of batches to process (None = all).
        include_hypothesis: Whether to ask the model to generate a FOL hypothesis.
        sample_filter: Optional predicate to filter samples before extraction.
    """
    client = OpenAI()
    samples_any = load_json_or_jsonl(in_path)

    samples: list[dict[str, Any]] = []
    for i, s in enumerate(samples_any):
        if not isinstance(s, dict):
            raise ValueError(f"Each sample must be a dict/object. Found {type(s)} at index {i}.")
        if sample_filter is not None and not sample_filter(s):
            continue
        samples.append(s)

    if not samples:
        console.print("No samples found.")
        return

    # Build all jobs
    jobs: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        jobs.append(
            {
                "index": idx,
                "id": sample.get("id"),
                "input_text": text_extractor(sample),
                "data": sample,
            }
        )

    # Determine total items to process
    total_items = len(jobs)
    if num_test_batches is not None:
        total_items = min(num_test_batches * batch_size, total_items)
    jobs = jobs[:total_items]

    # Resume: load existing valid records and skip already-processed items
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    existing = load_valid_records(out_file)
    existing_count = len(existing)

    if existing_count >= len(jobs):
        console.print(f"All {existing_count} items already processed in {out_path}")
        return

    if existing_count > 0:
        # Rewrite file to discard any truncated trailing record
        with out_file.open("w", encoding="utf-8") as f:
            for rec in existing:
                f.write(json.dumps(rec, ensure_ascii=False, indent=4) + "\n\n")
        console.print(f"Resuming: {existing_count}/{len(jobs)} items already done, continuing...")

    remaining_jobs = jobs[existing_count:]
    batches = chunked(remaining_jobs, batch_size)

    written = existing_count
    with (
        out_file.open("a" if existing_count > 0 else "w", encoding="utf-8") as f,
        Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress,
    ):
        task = progress.add_task("Extracting logic", total=len(jobs), completed=existing_count)

        for batch_jobs in batches:
            batch_texts = [j["input_text"] for j in batch_jobs]

            kb_list = extract_logic_batch(
                client, batch_texts, model=model, include_hypothesis=include_hypothesis
            )
            if len(kb_list) != len(batch_jobs):
                raise RuntimeError(
                    f"Model returned {len(kb_list)} items but expected {len(batch_jobs)}."
                )

            for job, kb in zip(batch_jobs, kb_list):
                record = {
                    "index": job["index"],
                    "id": job["id"],
                    "logic_kb": kb.model_dump()
                    if include_hypothesis
                    else kb.model_dump(exclude={"fol": {"hypothesis"}, "prolog": {"hypothesis"}}),
                    "data": job["data"],
                }
                f.write(json.dumps(record, ensure_ascii=False, indent=4) + "\n\n")
                written += 1

            f.flush()
            progress.advance(task, len(batch_jobs))

    console.print(f"Saved {written} items to {out_path}")


# ---------------------------------------------------------------------------
# LLM batch call
# ---------------------------------------------------------------------------


def extract_logic_batch(
    client: OpenAI,
    texts: list[str],
    model: str = "gpt-5-mini",
    *,
    include_hypothesis: bool = True,
) -> list[LogicKB]:
    """
    Extract logic knowledge bases from a batch of texts.

    Args:
        client: OpenAI client instance
        texts: List of input texts to process
        model: Model name to use
        include_hypothesis: Whether to ask the model to generate a FOL hypothesis.

    Returns:
        List of LogicKB objects
    """
    # Use clear delimiters so multi-line texts don't blur item boundaries
    parts = [f"=== ITEM [{i}] ===\n{t}" for i, t in enumerate(texts)]
    indexed_texts = "\n\n".join(parts)

    # Escape braces in USER_TEMPLATE so {input_text} doesn't get consumed by Python .format
    safe_user_template = USER_TEMPLATE.replace("{", "{{").replace("}", "}}").strip()

    user_msg = BATCH_USER_TEMPLATE.format(
        single_user_template=safe_user_template,
        indexed_texts=indexed_texts,
    )

    system_prompt = SYSTEM_PROMPT if include_hypothesis else SYSTEM_PROMPT_NO_HYPOTHESIS

    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        text_format=LogicKBList,
    )
    parsed = resp.output_parsed
    if parsed is None:
        raise RuntimeError("Model returned no parsed output.")
    return parsed.items
