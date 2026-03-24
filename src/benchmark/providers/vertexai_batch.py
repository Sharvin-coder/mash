"""Vertex AI batch provider using server-side BatchPredictionJob + GCS."""

import asyncio
import json
import os
import time
from typing import Any

from google.cloud import aiplatform, storage
from google.cloud.aiplatform_v1.types import JobState

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
from benchmark.utils import get_vertex_credentials, get_vertex_project_id, parse_jsonl

VERTEX_BATCH_LOG_PREFIX = "[Vertex Batch]"
OPENAI_SUCCESS_FINISH_REASONS = {"stop", "length"}

_RUNNING_STATES = {
    JobState.JOB_STATE_QUEUED,
    JobState.JOB_STATE_PENDING,
    JobState.JOB_STATE_RUNNING,
    JobState.JOB_STATE_CANCELLING,
    JobState.JOB_STATE_UPDATING,
}

_FAILED_STATES = {
    JobState.JOB_STATE_FAILED,
    JobState.JOB_STATE_CANCELLED,
    JobState.JOB_STATE_EXPIRED,
}


def _to_publisher_model_name(name: str) -> str:
    """Convert 'publisher/model' to 'publishers/publisher/models/model'.

    Already-qualified names and bare names (no slash) are returned as-is.
    """
    if name.startswith("publishers/"):
        return name
    parts = name.split("/", 1)
    if len(parts) == 2:
        return f"publishers/{parts[0]}/models/{parts[1]}"
    return name


def _build_request(item: BatchWorkItem) -> dict[str, Any]:
    """Build OpenAI-compatible batch request dict, stripping location from api_params."""
    body: dict[str, Any] = {
        "model": item["model"].name,
        "messages": [
            {"role": "system", "content": item["system_prompt"]},
            {"role": "user", "content": item["user_message"]},
        ],
    }
    if item["model"].api_params:
        for key, value in item["model"].api_params.items():
            if key == "location":
                continue
            body.setdefault(key, value)

    return {
        "custom_id": item["request_id"],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def _parse_openai_result(
    result: dict[str, Any],
) -> tuple[str | None, dict[str, Any], str | None, dict[str, Any]]:
    """Extract error, raw body, response text, and top-level raw from a batch result.

    Handles two formats:
    - OpenAI batch format: response has "status_code" + "body" keys, error is a dict
    - Vertex AI MaaS flat format: response is the chat completion object directly
      (has "choices" at top level), error is a string ("" for success)
    """
    response_data = result.get("response") or {}
    error_data = result.get("error")
    top_level_raw = response_data or error_data or {}

    # Top-level error — dict (OpenAI format) or non-empty string (Vertex MaaS format)
    if error_data:
        if isinstance(error_data, dict):
            code = error_data.get("code", "unknown")
            msg = error_data.get("message", "No error message")
            return f"{code}: {msg}", {}, None, top_level_raw
        elif isinstance(error_data, str):
            return error_data, {}, None, top_level_raw

    # Vertex AI MaaS flat format: chat completion object is directly in response_data
    if "choices" in response_data:
        body = response_data
    else:
        # OpenAI batch format: body is nested under a "body" key
        body = response_data.get("body")
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except json.JSONDecodeError:
                body = {}
        body = body or {}

        # HTTP error
        if response_data.get("status_code") != 200:
            err = body.get("error") or {}
            parts = [f"HTTP {response_data.get('status_code')}"]
            if err.get("code"):
                parts.append(f"code={err['code']}")
            if err.get("type"):
                parts.append(f"type={err['type']}")
            parts.append(err.get("message", "Unknown error"))
            return ": ".join(parts), {}, None, top_level_raw

    # Success — parse choices
    choices = body.get("choices", [])
    if not choices:
        return "No choices in response", {}, None, top_level_raw

    choice = choices[0]
    message = choice.get("message") or {}

    if message.get("refusal"):
        return f"Model refused: {message['refusal']}", body, None, top_level_raw

    finish_reason = choice.get("finish_reason")
    if finish_reason and finish_reason not in OPENAI_SUCCESS_FINISH_REASONS:
        return f"Unsuccessful finish_reason: {finish_reason}", body, None, top_level_raw

    return None, body, message.get("content", ""), top_level_raw


def _convert_from_openai_format(
    openai_results: list[dict[str, Any]],
) -> list[BatchResult]:
    """Convert OpenAI-format batch output lines to BatchResult list."""
    batch_results: list[BatchResult] = []

    for result in openai_results:
        request_id = result.get("custom_id", "")
        if not request_id:
            print(f"{VERTEX_BATCH_LOG_PREFIX} Skipping result with no custom_id")
            continue

        error, raw_body, response_text, top_level_raw = _parse_openai_result(result)

        generation_payload = None
        if error is None:
            generation_payload = {
                "response": response_text or "",
                "raw_api_response": raw_body,
            }

        batch_results.append(
            {
                "request_id": request_id,
                "error": error,
                "raw_api_response": top_level_raw,
                "generation": generation_payload,
                "judge": None,
            }
        )

    return batch_results


class VertexAIBatchProvider:
    """True Vertex AI batch provider using server-side BatchPredictionJob + GCS.

    Submits batch prediction jobs to Vertex AI and stores input/output on GCS.
    Jobs persist server-side, so progress survives process restarts.
    """

    def __init__(self) -> None:
        self._credentials = get_vertex_credentials()
        self._project_id = get_vertex_project_id()

        bucket_name = os.getenv("VERTEXAI_BATCH_GCS_BUCKET")
        if not bucket_name:
            bucket_name = f"{self._project_id}-benchmark-batch"
        self._bucket_name = bucket_name

        self._storage_client = storage.Client(
            project=self._project_id,
            credentials=self._credentials,
        )
        self._bucket_ensured = False

    async def _ensure_bucket(self) -> None:
        """Create GCS bucket if it doesn't exist (lazy, called once on first submit)."""
        if self._bucket_ensured:
            return

        storage_client = self._storage_client
        bucket_name = self._bucket_name

        def _check_and_create():
            bucket = storage_client.bucket(bucket_name)
            if not bucket.exists():
                print(
                    f"{VERTEX_BATCH_LOG_PREFIX} Creating GCS bucket: {bucket_name}"
                )
                storage_client.create_bucket(bucket_name, location="us")

        await asyncio.to_thread(_check_and_create)
        self._bucket_ensured = True

    async def submit(self, work_items: list[BatchWorkItem]) -> BatchSubmitResult:
        if not work_items:
            raise FatalBenchmarkError(
                "Vertex batch submit called with empty work_items"
            )

        first_item = work_items[0]
        location = (first_item["model"].api_params or {}).get("location")
        if not location or location == "global":
            raise FatalBenchmarkError(
                "Vertex AI batch API requires a regional location (e.g. 'us-central1'), "
                "not 'global'. Set 'location' in model api_params."
            )

        model_name = first_item["model"].name
        publisher_model = _to_publisher_model_name(model_name)

        await self._ensure_bucket()

        # Build JSONL input
        requests = [_build_request(item) for item in work_items]
        jsonl_content = "\n".join(json.dumps(req) for req in requests) + "\n"

        timestamp = int(time.time())
        safe_model = model_name.replace("/", "-")
        input_blob = f"batch-inputs/vertex-batch-{safe_model}-{timestamp}.jsonl"
        input_uri = f"gs://{self._bucket_name}/{input_blob}"
        output_prefix = (
            f"gs://{self._bucket_name}/batch-outputs/"
            f"vertex-batch-{safe_model}-{timestamp}/"
        )

        credentials = self._credentials
        project_id = self._project_id
        storage_client = self._storage_client
        bucket_name = self._bucket_name

        def _upload_and_submit() -> str:
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(input_blob)
            blob.upload_from_string(jsonl_content, content_type="application/jsonl")
            print(
                f"{VERTEX_BATCH_LOG_PREFIX} Uploaded {len(requests)} requests to {input_uri}"
            )

            job = aiplatform.BatchPredictionJob.submit(
                model_name=publisher_model,
                instances_format="jsonl",
                predictions_format="jsonl",
                gcs_source=input_uri,
                gcs_destination_prefix=output_prefix,
                project=project_id,
                location=location,
                credentials=credentials,
            )
            return job.resource_name

        resource_name = await asyncio.to_thread(_upload_and_submit)

        job_info: BatchJobInfo = {
            "job_id": resource_name,
            "provider": first_item["model"].provider,
            "status": "submitted",
            "model_name": model_name,
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metadata": {
                "location": location,
                "project_id": self._project_id,
                "input_uri": input_uri,
                "output_prefix": output_prefix,
                "publisher_model": publisher_model,
            },
        }

        print(f"{VERTEX_BATCH_LOG_PREFIX} Submitted job: {resource_name}")
        return {"job_info": job_info, "submitted_count": len(work_items)}

    async def poll(self, job_info: BatchJobInfo) -> BatchPollResult:
        resource_name = job_info["job_id"]
        location = job_info.get("metadata", {}).get("location", "us-central1")
        credentials = self._credentials
        project_id = self._project_id

        def _get_job_state() -> (
            tuple[JobState, str | None, str | None]
        ):
            job = aiplatform.BatchPredictionJob(
                batch_prediction_job_name=resource_name,
                project=project_id,
                location=location,
                credentials=credentials,
            )
            state = job.state
            output_dir = None
            if hasattr(job, "output_info") and job.output_info:
                output_dir = getattr(
                    job.output_info, "gcs_output_directory", None
                )
            error_msg = None
            if hasattr(job, "error") and job.error:
                error_msg = str(job.error)
            return state, output_dir, error_msg

        state, output_dir, error_msg = await asyncio.to_thread(_get_job_state)

        if state in _RUNNING_STATES:
            return {
                "status": BatchStatus.RUNNING,
                "completed_count": None,
                "results": None,
            }

        if state in _FAILED_STATES:
            if error_msg:
                print(f"{VERTEX_BATCH_LOG_PREFIX} Job failed: {error_msg}")
            return {
                "status": BatchStatus.FAILED,
                "completed_count": None,
                "results": None,
            }

        if state == JobState.JOB_STATE_SUCCEEDED:
            # Prefer the actual output directory from the job over our stored prefix
            gcs_path = output_dir or job_info.get("metadata", {}).get(
                "output_prefix", ""
            )
            results = await self._download_results(gcs_path)
            if results:
                return {
                    "status": BatchStatus.COMPLETED,
                    "completed_count": len(results),
                    "results": results,
                }
            print(
                f"{VERTEX_BATCH_LOG_PREFIX} Job succeeded but no results found at {gcs_path}"
            )
            return {
                "status": BatchStatus.FAILED,
                "completed_count": None,
                "results": None,
            }

        # Unknown state — treat as running to avoid premature failure
        print(f"{VERTEX_BATCH_LOG_PREFIX} Unknown job state: {state}")
        return {
            "status": BatchStatus.RUNNING,
            "completed_count": None,
            "results": None,
        }

    async def _download_results(self, gcs_path: str) -> list[BatchResult]:
        """Download and parse output JSONL files from GCS."""
        if not gcs_path:
            return []

        storage_client = self._storage_client

        def _download() -> list[dict[str, Any]]:
            if not gcs_path.startswith("gs://"):
                return []
            path = gcs_path[5:]  # strip "gs://"
            bucket_name, _, prefix = path.partition("/")
            bucket_obj = storage_client.bucket(bucket_name)
            blobs = list(bucket_obj.list_blobs(prefix=prefix))

            all_results: list[dict[str, Any]] = []
            for blob in blobs:
                if blob.name.endswith(".jsonl"):
                    content = blob.download_as_text()
                    all_results.extend(parse_jsonl(content))
            return all_results

        raw_results = await asyncio.to_thread(_download)
        return _convert_from_openai_format(raw_results)

    async def cancel(self, job_info: BatchJobInfo) -> BatchCancelResult:
        resource_name = job_info["job_id"]
        location = job_info.get("metadata", {}).get("location", "us-central1")
        credentials = self._credentials
        project_id = self._project_id

        try:

            def _cancel():
                job = aiplatform.BatchPredictionJob(
                    batch_prediction_job_name=resource_name,
                    project=project_id,
                    location=location,
                    credentials=credentials,
                )
                job.cancel()

            await asyncio.to_thread(_cancel)
            return {
                "success": True,
                "message": f"Batch {resource_name} cancellation initiated",
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to cancel batch {resource_name}: {e}",
            }
