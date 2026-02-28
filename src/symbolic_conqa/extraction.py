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
from .prompts import (
    BATCH_USER_TEMPLATE,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_NO_HYPOTHESIS,
    USER_TEMPLATE,
    USER_TEMPLATE_WITH_CONTEXT_PREDICATES,
)

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
    context_predicates_by_key: dict[str, str] | None = None,
    context_constants_by_key: dict[str, str] | None = None,
    context_rules_by_key: dict[str, str] | None = None,
    sample_key_fn: Callable[[dict[str, Any]], str] | None = None,
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
        context_predicates_by_key: Mapping from sample key to formatted predicates string.
        context_constants_by_key: Mapping from sample key to formatted constants string.
        context_rules_by_key: Mapping from sample key to formatted context rules string.
        sample_key_fn: Function to extract the lookup key from a sample dict.
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
        job: dict[str, Any] = {
            "index": idx,
            "id": sample.get("id"),
            "input_text": text_extractor(sample),
            "data": sample,
        }
        if sample_key_fn is not None:
            key = sample_key_fn(sample)
            if context_predicates_by_key is not None:
                job["context_predicates"] = context_predicates_by_key.get(key, "")
            if context_constants_by_key is not None:
                job["context_constants"] = context_constants_by_key.get(key, "")
            if context_rules_by_key is not None:
                job["context_rules"] = context_rules_by_key.get(key, "")
        jobs.append(job)

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

            # Collect per-item context predicates and constants if available
            ctx_preds: list[str] | None = None
            if context_predicates_by_key is not None:
                ctx_preds = [j.get("context_predicates", "") for j in batch_jobs]

            ctx_consts: list[str] | None = None
            if context_constants_by_key is not None:
                ctx_consts = [j.get("context_constants", "") for j in batch_jobs]

            ctx_rules: list[str] | None = None
            if context_rules_by_key is not None:
                ctx_rules = [j.get("context_rules", "") for j in batch_jobs]

            try:
                kb_list = extract_logic_batch(
                    client,
                    batch_texts,
                    model=model,
                    include_hypothesis=include_hypothesis,
                    context_predicates_per_item=ctx_preds,
                    context_constants_per_item=ctx_consts,
                    context_rules_per_item=ctx_rules,
                )
                if len(kb_list) != len(batch_jobs):
                    raise ValueError(
                        f"Model returned {len(kb_list)} items but expected {len(batch_jobs)}."
                    )
            except (ValueError, Exception) as e:
                console.print(f"[yellow]Batch failed ({e}), retrying items one-by-one...[/yellow]")
                kb_list = []
                for i, text in enumerate(batch_texts):
                    single_preds = [ctx_preds[i]] if ctx_preds else None
                    single_consts = [ctx_consts[i]] if ctx_consts else None
                    single_rules = [ctx_rules[i]] if ctx_rules else None
                    single_result = extract_logic_batch(
                        client,
                        [text],
                        model=model,
                        include_hypothesis=include_hypothesis,
                        context_predicates_per_item=single_preds,
                        context_constants_per_item=single_consts,
                        context_rules_per_item=single_rules,
                    )
                    kb_list.extend(single_result)

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
    context_predicates_per_item: list[str] | None = None,
    context_constants_per_item: list[str] | None = None,
    context_rules_per_item: list[str] | None = None,
) -> list[LogicKB]:
    """
    Extract logic knowledge bases from a batch of texts.

    Args:
        client: OpenAI client instance
        texts: List of input texts to process
        model: Model name to use
        include_hypothesis: Whether to ask the model to generate a FOL hypothesis.
        context_predicates_per_item: Optional per-item context predicates strings.
        context_constants_per_item: Optional per-item context constants strings.
        context_rules_per_item: Optional per-item context rules strings.

    Returns:
        List of LogicKB objects
    """
    has_context = (
        context_predicates_per_item is not None
        or context_constants_per_item is not None
        or context_rules_per_item is not None
    )

    # Build indexed text blocks
    if has_context:
        parts = []
        for i, t in enumerate(texts):
            ctx = (context_predicates_per_item[i] if context_predicates_per_item and i < len(context_predicates_per_item) else "")
            const = (context_constants_per_item[i] if context_constants_per_item and i < len(context_constants_per_item) else "")
            rules = (context_rules_per_item[i] if context_rules_per_item and i < len(context_rules_per_item) else "")
            block = f"=== ITEM [{i}] ===\nTEXT:\n{t}"
            if ctx:
                block += f"\n\nCONTEXT PREDICATES:\n{ctx}"
            if const:
                block += f"\n\nCONTEXT CONSTANTS (from the context document's KB):\n{const}"
            if rules:
                block += f"\n\nCONTEXT RULES (Prolog rules from the context document):\n{rules}"
            parts.append(block)
    else:
        parts = [f"=== ITEM [{i}] ===\n{t}" for i, t in enumerate(texts)]
    indexed_texts = "\n\n".join(parts)

    # Choose the appropriate user template
    if has_context:
        template = USER_TEMPLATE_WITH_CONTEXT_PREDICATES
    else:
        template = USER_TEMPLATE

    # Escape braces so {input_text}/{context_predicates} don't get consumed by Python .format
    safe_user_template = template.replace("{", "{{").replace("}", "}}").strip()

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
