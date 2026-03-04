import time
import asyncio
from typing import Any

from benchmark.exceptions import FatalBenchmarkError
from benchmark.protocols import (
    BatchCancelResult,
    BatchJobInfo,
    BatchPollResult,
    BatchResult,
    BatchStatus,
    BatchSubmitResult,
    BatchWorkItem,
)
from benchmark.types import GenerateResult
from benchmark.utils import (
    get_vertex_ai_client,
    openai_compat_generate,
    strip_reasoning_tags,
)

VERTEX_BATCH_LOG_PREFIX = "[Vertex Batch]"
_jobs: dict[str, dict[str, Any]] = {}


def _normalize_model_and_location(item: BatchWorkItem):
    model = item["model"]
    location = None
    if model.api_params and "location" in model.api_params:
        location = model.api_params["location"]
        model = model.model_copy(
            update={"api_params": {k: v for k, v in model.api_params.items() if k != "location"}}
        )
    return location, model


async def _run_one(item: BatchWorkItem) -> BatchResult:
    request_id = item["request_id"]
    system_prompt = item["system_prompt"] or ""
    user_message = item["user_message"]
    location, model = _normalize_model_and_location(item)

    try:
        async with get_vertex_ai_client(location) as client:
            result: GenerateResult = await openai_compat_generate(client, model, system_prompt, user_message)

        cleaned, reasoning = strip_reasoning_tags(result["response"])
        if reasoning is not None:
            result["raw_api_response"]["extracted_reasoning_content"] = reasoning
            result["response"] = cleaned

        return {
            "request_id": request_id,
            "error": None,
            "raw_api_response": result.get("raw_api_response"),
            "generation": result,
            "judge": None,
        }
    except Exception as e:
        return {
            "request_id": request_id,
            "error": str(e),
            "raw_api_response": None,
            "generation": None,
            "judge": None,
        }


async def _run_all(job_id: str, work_items: list[BatchWorkItem], max_concurrency: int):
    sem = asyncio.Semaphore(max_concurrency)

    async def _guarded(item: BatchWorkItem) -> BatchResult:
        async with sem:
            return await _run_one(item)

    tasks = [asyncio.create_task(_guarded(item)) for item in work_items]
    _jobs[job_id]["tasks"] = tasks

    try:
        results = await asyncio.gather(*tasks, return_exceptions=False)
        _jobs[job_id]["status"] = BatchStatus.COMPLETED
        _jobs[job_id]["results"] = results
    except asyncio.CancelledError:
        _jobs[job_id]["status"] = BatchStatus.CANCELLED
        _jobs[job_id]["results"] = None
        raise
    except Exception:
        _jobs[job_id]["status"] = BatchStatus.FAILED
        _jobs[job_id]["results"] = None


class VertexAIBatchProvider:
    def __init__(self, max_concurrency: int = 8):
        self.max_concurrency = max_concurrency

    async def submit(self, work_items: list[BatchWorkItem]) -> BatchSubmitResult:
        if not work_items:
            raise FatalBenchmarkError("Vertex batch submit called with empty work_items")

        model_name = work_items[0]["model"].name
        job_id = f"vertex-batch-{int(time.time())}-{id(work_items)}"

        _jobs[job_id] = {"status": BatchStatus.RUNNING, "results": None, "tasks": []}

        asyncio.create_task(_run_all(job_id, work_items, self.max_concurrency))

        print(f"{VERTEX_BATCH_LOG_PREFIX} Created job: {job_id}")

        return {
            "job_info": {
                "job_id": job_id,
                "provider": "vertexai",
                "status": "submitted",
                "model_name": model_name,
                "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "metadata": {"mode": "concurrent_openai_compat"},
            },
            "submitted_count": len(work_items),
        }

    async def poll(self, job_info: BatchJobInfo) -> BatchPollResult:
        job_id = job_info["job_id"]
        job = _jobs.get(job_id)

        if not job:
            return {"status": BatchStatus.FAILED, "completed_count": None, "results": None}

        status: BatchStatus = job["status"]

        if status == BatchStatus.RUNNING:
            return {"status": BatchStatus.RUNNING, "completed_count": None, "results": None}

        if status in (BatchStatus.CANCELLED, BatchStatus.FAILED):
            return {"status": BatchStatus.FAILED, "completed_count": None, "results": None}

        results: list[BatchResult] = job["results"] or []
        return {"status": BatchStatus.COMPLETED, "completed_count": len(results), "results": results}

    async def cancel(self, job_info: BatchJobInfo) -> BatchCancelResult:
        job_id = job_info["job_id"]
        job = _jobs.get(job_id)
        if not job:
            return {"success": False, "message": f"Batch {job_id} not found"}

        tasks: list[asyncio.Task] = job.get("tasks", [])
        for t in tasks:
            if not t.done():
                t.cancel()

        job["status"] = BatchStatus.CANCELLED
        return {"success": True, "message": f"Batch {job_id} cancelled successfully"}