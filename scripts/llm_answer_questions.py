#!/usr/bin/env python3
"""LLM baseline: answer questions in all_SQ_with_logic_yesno.jsonl using
context from all_context_with_logic.jsonl via a local HuggingFace model.

For each question the script:
  1. Looks up the matching context by ``data.url``.
  2. Strips HTML tags from context contents and concatenates them.
  3. Builds a chat prompt: system instruction + context + scenario + question.
  4. Runs the local LLM and extracts a yes / no / not-answerable prediction.
  5. Writes results to an output JSONL file and prints a summary.

Requires transformers and torch:
    pip install transformers torch accelerate

Usage:
    python scripts/llm_answer_questions.py
    python scripts/llm_answer_questions.py \\
        -m /mnt/raid0hdd1/liang/models/Llama-3.2-1B-Instruct \\
        -s results/all_SQ_with_logic_yesno.jsonl \\
        -c results/all_context_with_logic.jsonl \\
        -o results/llm_answers.jsonl \\
        --few-shot
"""

# NOTE: Do NOT add `from __future__ import annotations` here.
# Typer inspects annotations at runtime.

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

from symbolic_conqa.io_utils import load_json_or_jsonl, write_jsonl

console = Console()
app = typer.Typer(help="LLM baseline for yes/no question answering.")

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    """Remove HTML tags and collapse whitespace."""
    return re.sub(r"\s+", " ", _TAG_RE.sub(" ", text)).strip()


def _build_context_text(contents: list | str, max_chars: int = 3000) -> str:
    """Join and truncate context contents to *max_chars* characters."""
    if isinstance(contents, list):
        paragraphs = [_strip_html(p) for p in contents if p.strip()]
        text = " ".join(paragraphs)
    else:
        text = _strip_html(str(contents))
    return text[:max_chars]


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on provided "
    "government policy context. "
    "Read the context carefully, then answer the question with exactly one word: "
    "'yes', 'no', or 'not_answerable'. "
    "Do not add any other explanation."
)

# Few-shot examples covering all three answer types.
# Each entry: (context_snippet, scenario, question, answer)
_FEW_SHOT_EXAMPLES: list[tuple[str, str, str, str]] = [
    (
        # yes — claimant qualifies under the stated eligibility rule
        "You can claim Child Benefit if you are responsible for bringing up a "
        "child who is under 16, or under 20 if they stay in approved education "
        "or training. Only one person can claim Child Benefit for a child.",
        "I am the main carer for my 13-year-old son who lives with me full time.",
        "Can I claim Child Benefit for my son?",
        "yes",
    ),
    (
        # no — the rule explicitly excludes the described situation
        "You can get a full UK driving licence when you are 17 or over. "
        "You must be at least 17 years old to drive a car on public roads. "
        "Driving under the minimum age is illegal even with a licence.",
        "I am 15 years old and have recently passed my theory test.",
        "Can I start driving a car on public roads now?",
        "no",
    ),
    (
        # not_answerable — the context does not provide enough information
        "Council Tax Support helps people on low incomes pay their Council Tax. "
        "The amount of support you get depends on where you live, your household "
        "income, savings, and personal circumstances. Each local council sets "
        "its own rules about who qualifies and how much they can receive.",
        "I recently lost my job and moved to a new city. My savings are below £1,000.",
        "Will I receive exactly £200 per month in Council Tax Support?",
        "not_answerable",
    ),
]


def _few_shot_turns() -> list[dict]:
    """Return alternating user/assistant messages for the three few-shot examples."""
    turns: list[dict] = []
    for ctx, scenario, question, answer in _FEW_SHOT_EXAMPLES:
        user_content = (
            f"Context:\n{ctx}\n\n"
            f"Scenario: {scenario}\n\n"
            f"Question: {question}\n\n"
            "Answer with exactly one word — yes, no, or not_answerable:"
        )
        turns.append({"role": "user", "content": user_content})
        turns.append({"role": "assistant", "content": answer})
    return turns


def _build_messages(
    context_text: str,
    scenario: str,
    question: str,
    use_few_shot: bool = False,
) -> list[dict]:
    """Return a chat messages list for the model's apply_chat_template.

    When *use_few_shot* is True, three labelled (user, assistant) turns are
    prepended before the actual question so the model can learn the expected
    output format from in-context examples.
    """
    user_content = (
        f"Context:\n{context_text}\n\n"
        f"Scenario: {scenario}\n\n"
        f"Question: {question}\n\n"
        "Answer with exactly one word — yes, no, or not_answerable:"
    )
    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if use_few_shot:
        messages.extend(_few_shot_turns())
    messages.append({"role": "user", "content": user_content})
    return messages


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

def _parse_prediction(generated_text: str) -> str:
    """Extract yes / no / not_answerable from raw model output."""
    text = generated_text.strip().lower()
    # Accept common variants
    if re.search(r"\bnot[_\s]answerable\b", text):
        return "not answerable"
    if text.startswith("yes") or re.search(r"\byes\b", text[:60]):
        return "yes"
    if text.startswith("no") or re.search(r"\bno\b", text[:60]):
        return "no"
    # Fall back to first token
    first = text.split()[0] if text.split() else ""
    if "yes" in first:
        return "yes"
    if "no" in first:
        return "no"
    return "not answerable"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _question_type(sq_data: dict) -> str:
    if sq_data.get("not_answerable"):
        return "not_answerable"
    gold = [a[0].strip().lower() for a in sq_data.get("answers", [])]
    if gold and all(v in ("yes", "no") for v in gold):
        return "yesno"
    return "value"


def _gold_yn(answers: list) -> str:
    vals = [a[0].strip().lower() for a in answers]
    return "yes" if "yes" in vals else "no"


def _print_summary(results: list[dict]) -> None:
    total = len(results)
    no_ctx = sum(1 for r in results if not r["context_found"])
    errors = sum(1 for r in results if r.get("error"))

    console.print(f"\n[bold]── Overall ({total} questions) ──[/bold]")
    console.print(f"  Context match  : {total - no_ctx} / {total}")
    console.print(f"  Errors         : {errors}")

    yesno = [r for r in results if r["question_type"] == "yesno"]
    if yesno:
        correct = sum(1 for r in yesno if r["predicted"] == _gold_yn(r["gold_answers"]))
        console.print(
            f"\n[bold]Yes/No questions[/bold] ({len(yesno)}): "
            f"{correct} / {len(yesno)} correct "
            f"({100 * correct // len(yesno)}%)"
        )
        yes_pred = sum(1 for r in yesno if r["predicted"] == "yes")
        no_pred  = sum(1 for r in yesno if r["predicted"] == "no")
        na_pred  = sum(1 for r in yesno if r["predicted"] == "not answerable")
        console.print(
            f"  Predicted: yes={yes_pred}  no={no_pred}  not_answerable={na_pred}"
        )

    na = [r for r in results if r["question_type"] == "not_answerable"]
    if na:
        correct_na = sum(1 for r in na if r["predicted"] == "not answerable")
        console.print(
            f"\n[bold]Not-answerable[/bold] ({len(na)}): "
            f"{correct_na} / {len(na)} correctly predicted"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@app.command()
def main(
    model_path: str = typer.Option(
        "/mnt/raid0hdd1/liang/models/Llama-3.1-8B-Instruct",
        "-m",
        "--model",
        help="Path to local HuggingFace model directory",
    ),
    sq_path: str = typer.Option(
        "results/all_SQ_with_logic_yesno.jsonl",
        "-s",
        "--sq",
        help="Question file (all_SQ_with_logic_yesno.jsonl)",
    ),
    context_path: str = typer.Option(
        "results/all_context_with_logic.jsonl",
        "-c",
        "--context",
        help="Context KB file (all_context_with_logic.jsonl)",
    ),
    output_path: str = typer.Option(
        "results/llm_answers.jsonl",
        "-o",
        "--output",
        help="Output path for answer results",
    ),
    max_new_tokens: int = typer.Option(
        16,
        "--max-new-tokens",
        help="Maximum tokens the model may generate (default 16)",
    ),
    max_context_chars: int = typer.Option(
        3000,
        "--max-context-chars",
        help="Maximum characters of context to include in prompt (default 3000)",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Device map for model loading: 'auto', 'cpu', 'cuda', 'cuda:0', …",
    ),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        help="Inference batch size (default 1)",
    ),
    few_shot: bool = typer.Option(
        False,
        "--few-shot/--no-few-shot",
        help=(
            "Prepend three labelled (yes / no / not_answerable) examples to "
            "each prompt so the model learns the output format in-context."
        ),
    ),
) -> None:
    """Answer yes/no questions with a local LLM using retrieved context."""

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    console.print(f"Loading model from [cyan]{model_path}[/cyan] ...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        console.print(
            "[red]Error:[/red] transformers and torch are required. "
            "Install with: pip install transformers torch accelerate"
        )
        raise typer.Exit(1)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        dtype=torch.float16 if device != "cpu" else torch.float32,
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    console.print("Model loaded.")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    sq_records = load_json_or_jsonl(sq_path)
    ctx_records = load_json_or_jsonl(context_path)
    ctx_index: dict[str, dict] = {
        r["data"]["url"]: r for r in ctx_records if r.get("data", {}).get("url")
    }
    console.print(f"Loaded {len(ctx_index)} context records (indexed by URL)")
    console.print(f"Loaded {len(sq_records)} question records")
    if few_shot:
        console.print(
            f"Few-shot: [cyan]{len(_FEW_SHOT_EXAMPLES)} examples[/cyan] "
            "prepended to each prompt"
        )

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    results: list[dict] = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Answering questions", total=len(sq_records))

        for sq_rec in sq_records:
            sq_data = sq_rec.get("data", {})
            url = sq_data.get("url", "")
            rec_id = sq_rec.get("id") or sq_data.get("id") or sq_rec.get("index")

            q_type = _question_type(sq_data)
            ctx_rec = ctx_index.get(url)
            context_found = ctx_rec is not None

            predicted = "not answerable"
            raw_output = ""
            error = None

            if not context_found:
                error = f"no context found for url: {url}"
            else:
                try:
                    ctx_contents = ctx_rec["data"].get("contents", "")
                    context_text = _build_context_text(ctx_contents, max_context_chars)
                    scenario = sq_data.get("scenario", "")
                    question = sq_data.get("question", "")

                    messages = _build_messages(
                        context_text, scenario, question, use_few_shot=few_shot
                    )
                    # Apply chat template (adds BOS, role tokens, etc.)
                    # transformers ≥5.x returns BatchEncoding; older returns tensor
                    encoded = tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    )
                    if hasattr(encoded, "input_ids"):
                        input_ids = encoded.input_ids.to(model.device)
                    else:
                        input_ids = encoded.to(model.device)

                    with torch.no_grad():
                        output_ids = model.generate(
                            input_ids,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                        )

                    # Decode only the newly generated tokens
                    new_tokens = output_ids[0][input_ids.shape[-1]:]
                    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
                    predicted = _parse_prediction(raw_output)

                except Exception as exc:
                    error = str(exc)
                    predicted = "not answerable"

            results.append({
                "id": rec_id,
                "url": url,
                "question": sq_data.get("question"),
                "scenario": sq_data.get("scenario"),
                "question_type": q_type,
                "context_found": context_found,
                "predicted": predicted,
                "raw_output": raw_output,
                "gold_answers": sq_data.get("answers", []),
                "not_answerable": sq_data.get("not_answerable"),
                "error": error,
            })
            progress.advance(task)

    write_jsonl(output_path, results)
    console.print(f"\nWritten {len(results)} answers to {output_path}")
    _print_summary(results)


if __name__ == "__main__":
    app()
