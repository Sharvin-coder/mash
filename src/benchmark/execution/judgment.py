"""Judgment executors for benchmark workflow."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Sequence

from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm.asyncio import tqdm

import os

from benchmark.checkpoint import (
    Checkpoint,
    CheckpointWriter,
    GenerationStatus,
    get_generation_status,
)
from benchmark.config import (
    BenchmarkConfig,
    FAILURE_TYPE_CIM,
    JUDGE_MODEL,
    JUDGE_MODEL_GEMINI,
    JUDGE_LOCATION,
    JUDGE_TEMPERATURE,
    JUDGE_MODEL_ENTRY_OPENROUTER,
    JUDGE_MODEL_OPENROUTER,
    ModelEntry,
)
from benchmark.exceptions import FatalBenchmarkError, NonRetryableError
from benchmark.prompts import (
    build_cim_judge_prompt,
    build_cim_official_judge_prompt,
    build_judge_prompt,
    get_cim_judge_prompt,
    get_judge_system_prompt,
)
from benchmark.providers.gemini import gemini_generate
from benchmark.providers.openrouter import openrouter_generate_response
from benchmark.types import JudgeResult
from benchmark.utils import (
    extract_json_from_response,
    get_max_retries,
    get_vertex_ai_client,
)
from benchmark.work_planner import InputEntry, WorkItem, ensure_entry_configuration

JUDGE_SUCCESS_FINISH_REASONS = {"stop", "length"}

# Runtime judge provider (set via set_judge_provider, takes precedence over env var)
_runtime_judge_provider: str | None = None
# Runtime judge model override (set via set_judge_model)
_runtime_judge_model: str | None = None
# Runtime CIM judge variant (set via set_cim_judge_variant)
_runtime_cim_judge_variant: str | None = None


def set_judge_provider(provider: str | None) -> None:
    """Set runtime judge provider (takes precedence over env var)."""
    global _runtime_judge_provider
    _runtime_judge_provider = provider


def get_judge_provider() -> str:
    """Get judge provider with precedence: runtime > env > default."""
    if _runtime_judge_provider is not None:
        return _runtime_judge_provider
    return os.getenv("JUDGE_PROVIDER", "gemini")


def set_judge_model(model_name: str | None) -> None:
    """Set runtime judge model override."""
    global _runtime_judge_model
    _runtime_judge_model = model_name


def get_judge_model() -> str:
    """Get judge model with precedence: runtime override > default."""
    if _runtime_judge_model is not None:
        return _runtime_judge_model
    provider = get_judge_provider()
    if provider == "openrouter":
        return JUDGE_MODEL_OPENROUTER
    if provider == "gemini":
        return JUDGE_MODEL_GEMINI
    return JUDGE_MODEL


def set_cim_judge_variant(variant: str | None) -> None:
    """Set runtime CIM judge variant."""
    global _runtime_cim_judge_variant
    _runtime_cim_judge_variant = variant


def get_cim_judge_variant() -> str:
    """Get CIM judge variant with precedence: runtime > default."""
    if _runtime_cim_judge_variant is not None:
        return _runtime_cim_judge_variant
    return "reveal_paper_compat"


__all__ = [
    "SequentialJudgmentExecutor",
    "build_judgment_tasks",
    "evaluate_with_judge",
    "get_cim_judge_variant",
    "get_judge_provider",
    "judge_response",
    "set_cim_judge_variant",
    "set_judge_model",
    "set_judge_provider",
]


@dataclass(slots=True)
class JudgmentTask:
    """Single judgment workload item."""

    entry: InputEntry
    model: ModelEntry
    gen_idx: int


def build_judgment_tasks(
    checkpoint: Checkpoint, work_items: Sequence[WorkItem]
) -> list[JudgmentTask]:
    """Filter work items for those needing judgment."""
    tasks: list[JudgmentTask] = []
    for entry, model, gen_idx in work_items:
        status = get_generation_status(
            checkpoint, entry["hash_id"], model.name, gen_idx
        )
        if status == GenerationStatus.NEEDS_JUDGE:
            tasks.append(JudgmentTask(entry=entry, model=model, gen_idx=gen_idx))
    return tasks


class SequentialJudgmentExecutor:
    """Executes judgment tasks sequentially with concurrency control."""

    def __init__(self) -> None:
        self._semaphore: asyncio.Semaphore | None = None

    async def run(
        self,
        tasks: Sequence[JudgmentTask],
        checkpoint: Checkpoint,
        config: BenchmarkConfig,
    ) -> None:
        if not tasks:
            return

        self._semaphore = asyncio.Semaphore(config.concurrency)
        checkpoint_writer = CheckpointWriter(checkpoint, config.output)

        success_count = 0
        error_count = 0
        count_lock = asyncio.Lock()

        async def _run_task(task: JudgmentTask, pbar: tqdm) -> None:
            nonlocal success_count, error_count
            assert self._semaphore is not None

            async with self._semaphore:
                await _process_judgment_task(
                    task,
                    checkpoint_writer,
                    store_raw_api_responses=config.store_raw_api_responses,
                )
                hash_id = task.entry["hash_id"]
                generation = checkpoint["entries"][hash_id]["results"][task.model.name][
                    "generations"
                ][task.gen_idx]
                async with count_lock:
                    if generation.get("error"):
                        error_count += 1
                    else:
                        success_count += 1
                    pbar.set_postfix_str(f"ok={success_count} err={error_count}")
                    pbar.update(1)

        with tqdm(total=len(tasks), desc="Evaluating responses") as pbar:
            try:
                await asyncio.gather(*(_run_task(task, pbar) for task in tasks))
            finally:
                await checkpoint_writer.close()


def _parse_judge_content(content: str, raw_api_response: dict) -> JudgeResult:
    """Extract score/reasoning from judge response, handling <think> blocks."""
    thinking_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    thinking_content = thinking_match.group(1).strip() if thinking_match else ""

    answer_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    judge_data = extract_json_from_response(answer_content)
    raw_api_response["extracted_reasoning_content"] = thinking_content
    score = judge_data.get("score") or judge_data.get("rating")
    if score is None:
        raise ValueError(f"Invalid judge data: {judge_data}")
    return JudgeResult(
        score=int(score),
        reasoning=judge_data["reasoning"],
        raw_api_response=raw_api_response,
    )


def _parse_cim_judge_content(
    content: str,
    raw_api_response: dict,
    entry: InputEntry,
) -> dict[str, Any]:
    """Parse CIM judge response and compute violation/completeness rates."""
    thinking_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    thinking_content = thinking_match.group(1).strip() if thinking_match else ""

    answer_content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    judge_data = extract_json_from_response(answer_content)
    raw_api_response["extracted_reasoning_content"] = thinking_content

    revealed = set(judge_data.get("revealed_attributes", []))
    used = set(judge_data.get("used_attributes", []))
    required = set(entry.get("required_attributes", []))
    forbidden = set(entry.get("forbidden_attributes", []))

    violation_rate = len(revealed & forbidden) / max(1, len(forbidden))
    completeness_rate = len(used & required) / max(1, len(required))

    return {
        "revealed_attributes": sorted(revealed),
        "used_attributes": sorted(used),
        "violation_rate": violation_rate,
        "completeness_rate": completeness_rate,
        "raw_api_response": raw_api_response,
    }


def _parse_cim_reveal_judge_content(
    content: str,
    raw_api_response: dict,
    entry: InputEntry,
) -> dict[str, Any]:
    """Parse CIM REVEAL judge response (per-attribute verdicts)."""
    thinking_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    thinking_content = thinking_match.group(1).strip() if thinking_match else ""

    answer_content = re.sub(
        r"<think>.*?</think>", "", content, flags=re.DOTALL
    ).strip()

    judge_data = extract_json_from_response(answer_content)
    raw_api_response["extracted_reasoning_content"] = thinking_content

    required = set(entry.get("required_attributes", []))
    forbidden = set(entry.get("forbidden_attributes", []))

    revealed: set[str] = set()
    used: set[str] = set()
    per_attribute: list[dict[str, Any]] = []
    for attr_result in judge_data.get("attributes", []):
        aid = attr_result["id"]
        if attr_result.get("revealed", False):
            revealed.add(aid)
        if attr_result.get("used", False):
            used.add(aid)
        per_attribute.append(attr_result)

    return {
        "revealed_attributes": sorted(revealed),
        "used_attributes": sorted(used),
        "violation_rate": len(revealed & forbidden) / max(1, len(forbidden)),
        "completeness_rate": len(used & required) / max(1, len(required)),
        "per_attribute": per_attribute,
        "raw_api_response": raw_api_response,
    }


def _parse_cim_official_reveal_content(
    content: str,
    raw_api_response: dict,
    entry: InputEntry,
) -> dict[str, Any]:
    """Parse official CIMemories REVEAL judge response (flat dict output).

    The official judge returns a flat JSON dict where keys are revealed attribute
    names and values are evidence sentences. Attributes not present as keys are
    considered not revealed.
    """
    thinking_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    thinking_content = thinking_match.group(1).strip() if thinking_match else ""

    answer_content = re.sub(
        r"<think>.*?</think>", "", content, flags=re.DOTALL
    ).strip()

    judge_data = extract_json_from_response(answer_content)
    raw_api_response["extracted_reasoning_content"] = thinking_content

    required = set(entry.get("required_attributes", []))
    forbidden = set(entry.get("forbidden_attributes", []))
    all_attrs = list(required | forbidden)

    revealed_dict: dict[str, int] = {}
    for attr in all_attrs:
        revealed_dict[attr] = 1 if attr in judge_data else 0

    revealed_set = {a for a, v in revealed_dict.items() if v == 1}
    violation_rate = len(revealed_set & forbidden) / max(1, len(forbidden))
    completeness_rate = len(revealed_set & required) / max(1, len(required))

    return {
        "revealed_attributes": revealed_dict,
        "violation_rate": violation_rate,
        "completeness_rate": completeness_rate,
        "revealed_attributes_explanation": judge_data,
        "raw_api_response": raw_api_response,
    }


async def _judge_via_vertexai(
    system_prompt: str, user_message: str, model_name: str | None = None
) -> JudgeResult:
    """Call Vertex AI judge."""
    from benchmark.config import BENCHMARK_SEED

    judge_model = model_name or JUDGE_MODEL
    async with get_vertex_ai_client(location=JUDGE_LOCATION) as client:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        api_params = {
            "model": judge_model,
            "messages": messages,
            "temperature": JUDGE_TEMPERATURE,
            "seed": BENCHMARK_SEED,
        }

        try:
            response = await client.chat.completions.create(**api_params)
        except Exception as e:
            raise RuntimeError(
                f"Vertex AI API call failed for judge model {judge_model}. "
                f"Error: {type(e).__name__}: {str(e)}. "
            ) from e

        choice = response.choices[0]
        finish_reason = choice.finish_reason
        if finish_reason and finish_reason not in JUDGE_SUCCESS_FINISH_REASONS:
            raise NonRetryableError(
                f"Judge unsuccessful finish_reason: {finish_reason}"
            )

        return _parse_judge_content(choice.message.content, response.model_dump())


async def _judge_via_openrouter(
    system_prompt: str, user_message: str, model_name: str | None = None
) -> JudgeResult:
    """Call OpenRouter judge with google-vertex provider routing."""
    if model_name:
        model_entry = ModelEntry(
            name=model_name,
            api_params=JUDGE_MODEL_ENTRY_OPENROUTER.api_params,
        )
    else:
        model_entry = JUDGE_MODEL_ENTRY_OPENROUTER
    gen_result = await openrouter_generate_response(
        model_entry, system_prompt, user_message
    )
    return _parse_judge_content(gen_result["response"], gen_result["raw_api_response"])


async def _judge_via_gemini(
    system_prompt: str, user_message: str, model_name: str | None = None
) -> JudgeResult:
    """Call Gemini (Google AI Studio) judge with structured JSON output."""
    from benchmark.config import BENCHMARK_SEED

    judge_model = model_name or JUDGE_MODEL_GEMINI
    model_entry = ModelEntry(
        name=judge_model,
        provider="gemini",
        api_params={
            "temperature": JUDGE_TEMPERATURE,
            "max_output_tokens": 4096,
            "response_mime_type": "application/json",
            "response_json_schema": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "score": {"type": "integer"},
                },
                "required": ["reasoning", "score"],
            },
        },
    )
    gen_result = await gemini_generate(model_entry, system_prompt, user_message)
    return _parse_judge_content(gen_result["response"], gen_result["raw_api_response"])


@retry(
    stop=stop_after_attempt(get_max_retries()),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_not_exception_type((FatalBenchmarkError, NonRetryableError)),
    reraise=True,
)
async def judge_response(
    system_prompt: str,
    user_message: str,
) -> JudgeResult:
    """Call judge via configured provider (Vertex AI or OpenRouter)."""
    provider = get_judge_provider()
    judge_model = get_judge_model()
    try:
        if provider == "openrouter":
            return await _judge_via_openrouter(
                system_prompt, user_message, model_name=judge_model
            )
        elif provider == "vertexai":
            return await _judge_via_vertexai(
                system_prompt, user_message, model_name=judge_model
            )
        elif provider == "gemini":
            return await _judge_via_gemini(
                system_prompt, user_message, model_name=judge_model
            )
        else:
            raise FatalBenchmarkError(
                f"Unknown judge provider: {provider}. "
                "Supported values: 'vertexai', 'openrouter', 'gemini'"
            )
    except (FatalBenchmarkError, NonRetryableError):
        raise
    except Exception as e:
        raise NonRetryableError(
            f"Failed to parse judge response: {type(e).__name__}: {e}"
        ) from e


@retry(
    stop=stop_after_attempt(get_max_retries()),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_not_exception_type((FatalBenchmarkError, NonRetryableError)),
    reraise=True,
)
async def judge_response_cim(
    system_prompt: str,
    user_message: str,
    entry: InputEntry,
) -> dict[str, Any]:
    """Call judge for CIM evaluation and return CIM-specific results."""
    provider = get_judge_provider()
    judge_model = get_judge_model()
    variant = get_cim_judge_variant()
    if variant == "reveal_official":
        parse_fn = _parse_cim_official_reveal_content
    elif variant == "reveal_paper_compat":
        parse_fn = _parse_cim_reveal_judge_content
    else:
        parse_fn = _parse_cim_judge_content
    try:
        if provider == "openrouter":
            if judge_model and judge_model != JUDGE_MODEL_OPENROUTER:
                model_entry = ModelEntry(
                    name=judge_model,
                    api_params=JUDGE_MODEL_ENTRY_OPENROUTER.api_params,
                )
            else:
                model_entry = JUDGE_MODEL_ENTRY_OPENROUTER
            gen_result = await openrouter_generate_response(
                model_entry, system_prompt, user_message
            )
            return parse_fn(
                gen_result["response"], gen_result["raw_api_response"], entry
            )
        elif provider == "vertexai":
            from benchmark.config import BENCHMARK_SEED

            model = judge_model or JUDGE_MODEL
            async with get_vertex_ai_client(location=JUDGE_LOCATION) as client:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ]
                response = await client.chat.completions.create(
                    model=model, messages=messages, temperature=JUDGE_TEMPERATURE,
                    seed=BENCHMARK_SEED,
                )
                return parse_fn(
                    response.choices[0].message.content,
                    response.model_dump(),
                    entry,
                )
        elif provider == "gemini":
            model_entry = ModelEntry(
                name=judge_model or JUDGE_MODEL_GEMINI,
                provider="gemini",
                api_params={
                    "temperature": JUDGE_TEMPERATURE,
                    "max_output_tokens": 4096,
                    "response_mime_type": "application/json",
                },
            )
            gen_result = await gemini_generate(
                model_entry, system_prompt, user_message
            )
            return parse_fn(
                gen_result["response"], gen_result["raw_api_response"], entry
            )
        else:
            raise FatalBenchmarkError(
                f"Unknown judge provider: {provider}. "
                "Supported values: 'vertexai', 'openrouter', 'gemini'"
            )
    except (FatalBenchmarkError, NonRetryableError):
        raise
    except Exception as e:
        raise NonRetryableError(
            f"Failed to parse CIM judge response: {type(e).__name__}: {e}"
        ) from e


async def evaluate_with_judge(
    entry: InputEntry,
    memory_response: str,
) -> tuple[JudgeResult | dict[str, Any] | None, str | None]:
    """Call internal judge to evaluate a response."""
    failure_type = ensure_entry_configuration(entry)

    if failure_type == FAILURE_TYPE_CIM:
        variant = get_cim_judge_variant()
        judge_system_prompt = get_cim_judge_prompt(variant)

        if variant == "reveal_official":
            judge_system_prompt = ""  # official uses user message only
            attr_map = entry.get("cim_metadata", {}).get("attribute_memory_map", {})
            judge_user_msg = build_cim_official_judge_prompt(
                attribute_memory_map=attr_map,
                memory_response=memory_response,
                required_attributes=entry.get("required_attributes", []),
                forbidden_attributes=entry.get("forbidden_attributes", []),
            )
        elif variant == "reveal_paper_compat":
            attr_map = entry.get("cim_metadata", {}).get("attribute_memory_map", {})
            judge_user_msg = build_cim_judge_prompt(
                memories=entry["memories"],
                query=entry["query"],
                memory_response=memory_response,
                attribute_memory_map=attr_map,
                required_attributes=entry.get("required_attributes", []),
                forbidden_attributes=entry.get("forbidden_attributes", []),
            )
        else:
            judge_user_msg = build_judge_prompt(
                memories=entry["memories"],
                query=entry["query"],
                memory_response=memory_response,
            )
    else:
        judge_system_prompt = get_judge_system_prompt(failure_type)
        judge_user_msg = build_judge_prompt(
            memories=entry["memories"],
            query=entry["query"],
            memory_response=memory_response,
        )

    try:
        if failure_type == FAILURE_TYPE_CIM:
            return await judge_response_cim(
                judge_system_prompt, judge_user_msg, entry
            ), None
        return await judge_response(judge_system_prompt, judge_user_msg), None
    except FatalBenchmarkError:
        raise
    except Exception as e:
        return None, f"Judge evaluation failed: {repr(e)}"


async def _process_judgment_task(
    task: JudgmentTask,
    checkpoint_writer: CheckpointWriter,
    store_raw_api_responses: bool,
) -> None:
    """Process judgment for an existing generation."""
    checkpoint = checkpoint_writer.checkpoint
    hash_id = task.entry["hash_id"]
    existing_generation = checkpoint["entries"][hash_id]["results"][task.model.name][
        "generations"
    ][task.gen_idx]

    try:
        memory_response = existing_generation.get("memory_response")
        if not memory_response:
            raise FatalBenchmarkError(
                f"Missing model response for {hash_id} {task.model.name} gen_idx={task.gen_idx}"
            )

        judge_result, judge_error = await evaluate_with_judge(
            task.entry, memory_response
        )

        def _apply(checkpoint: Checkpoint) -> None:
            generation = checkpoint["entries"][hash_id]["results"][task.model.name][
                "generations"
            ][task.gen_idx]
            if judge_error:
                generation["error"] = judge_error
                generation["judge"] = None
            else:
                assert judge_result is not None
                if not store_raw_api_responses:
                    if isinstance(judge_result, dict) and "raw_api_response" in judge_result:
                        judge_result["raw_api_response"] = {}
                generation["judge"] = judge_result
                generation["error"] = None

        await checkpoint_writer.update(_apply)

    except (asyncio.CancelledError, FatalBenchmarkError):
        raise
    except Exception as exc:

        def _apply_error(checkpoint: Checkpoint, err: Exception = exc) -> None:
            generation = checkpoint["entries"][hash_id]["results"][task.model.name][
                "generations"
            ][task.gen_idx]
            generation["error"] = f"Judgment error: {err}"
            generation["judge"] = None

        await checkpoint_writer.update(_apply_error)
