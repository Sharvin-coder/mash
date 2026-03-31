#!/usr/bin/env python3
"""Partition CIM memories into 11 categories.

Reads samples from the facebook/CIMemories HuggingFace dataset (full_profile
mode), partitions each sample's flat memory list into 11 categories using an
LLM, and writes a JSONL file compatible with the benchmark runner's partitioned
mode.

Each output row preserves all CIM-specific fields (required_attributes,
forbidden_attributes, cim_metadata) so the judge can evaluate properly.

─── HOW TO EDIT ───────────────────────────────────────────────────────────────
  • Change the model / location / temperature  →  MODEL block below
  • Change input/output paths                  →  RUN CONFIG below
  • Change concurrency or retry behaviour      →  RUN CONFIG below
  • Change what the LLM is told to do          →  SYSTEM_PROMPT below
───────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import os

os.environ.setdefault(
    "VERTEXAI_SERVICE_ACCOUNT_PATH",
    str(Path.home() / "Downloads" / "VERTEXAI_SERVICE_ACCOUNT.json"),
)

# ── MODEL ─────────────────────────────────────────────────────────────────────
MODEL_NAME     = "meta/llama-3.3-70b-instruct-maas"
MODEL_LOCATION = "us-central1"   # VertexAI region where the model is deployed
TEMPERATURE    = 0
# ──────────────────────────────────────────────────────────────────────────────

# ── RUN CONFIG ────────────────────────────────────────────────────────────────
CONCURRENCY     = 7    # max simultaneous API requests
MAX_RETRIES     = 5    # retry attempts per sample on parse / API failure
_PROJECT_ROOT   = Path(__file__).parent.parent.parent.parent  # repo root
CIM_DATASET_ID  = "facebook/CIMemories"
CIM_LABELS_FILE = _PROJECT_ROOT / "outputs/CIM/cim_labels.json"
OUTPUT_FILE     = _PROJECT_ROOT / "benchmark_samples/cim/partitioned/llama3p3_70b/cim.jsonl"
# ──────────────────────────────────────────────────────────────────────────────

# ── PROMPT ────────────────────────────────────────────────────────────────────
# Same 11 categories as partition_memories.py for consistency across datasets.
CATEGORIES = [
    "personal",
    "education",
    "employment",
    "finance",
    "housing",
    "legal",
    "health",
    "schedule",
    "identity",
    "social",
    "romantic",
]

SYSTEM_PROMPT = """\
You are a memory classifier. Your task is to sort a list of personal memories
into exactly one of the following categories:

health       – physical or mental health, medical conditions, treatments, medications, fitness, therapy
identity     – core personal identity traits such as nationality, religion, gender identity, values, beliefs
social       – non-romantic relationships and interactions with friends, family, acquaintances, or colleagues
romantic     – intimate or romantic relationships including dating, partners, marriage, attraction, breakups
personal     – hobbies, preferences, lifestyle choices, personality traits, interests
education    – schooling, degrees, courses, academic history, tutoring, learning experiences
employment   – jobs, work history, workplace experiences, colleagues, professional skills
finance      – money, savings, income, expenses, debt, investments, banking, taxes
housing      – home, residence, living situation, roommates, neighbors, rent, mortgage
legal        – legal issues, contracts, court matters, rights, criminal record, official documents
schedule     – appointments, routines, recurring events, time-based plans, daily habits

Rules:
1. Each memory must appear in exactly one category.
2. Do not drop or duplicate memories.
3. If a memory could fit multiple categories, choose the most specific category.
4. Categories with no memories must contain an empty list [].
5. Do not modify the memory text.

Return ONLY a single-line JSON object with the following keys in this exact order:

{"health": [...], "identity": [...], "social": [...], "romantic": [...], "personal": [...], "education": [...], "employment": [...], "finance": [...], "housing": [...], "legal": [...], "schedule": [...]}
"""
# ──────────────────────────────────────────────────────────────────────────────


# ── Internals (no need to edit below) ─────────────────────────────────────────

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # add src/ so 'benchmark' is importable

from benchmark.utils import extract_json_from_response, get_vertex_ai_client  # noqa: E402
from benchmark.datasets.cim import CIMDataset  # noqa: E402


def _load_checkpoint() -> set[str]:
    """Return the set of hash_ids already written to the output file."""
    done: set[str] = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["hash_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return done


def _validate_partition(
    memories: list[str], raw: dict
) -> dict[str, list[str]]:
    """Ensure every input memory appears exactly once in the result.

    Accepts whatever the model returned for each category, then appends any
    memories the model missed to 'personal' as a safe fallback.
    """
    result: dict[str, list[str]] = {cat: [] for cat in CATEGORIES}
    placed: set[str] = set()

    for cat in CATEGORIES:
        for mem in raw.get(cat, []):
            if mem in memories and mem not in placed:
                result[cat].append(mem)
                placed.add(mem)

    # Fallback: anything the model missed goes to 'personal'
    for mem in memories:
        if mem not in placed:
            result["personal"].append(mem)

    return result


async def _classify(
    client,
    memories: list[str],
    semaphore: asyncio.Semaphore,
) -> dict[str, list[str]]:
    """Call the LLM and return a validated 11-category partition."""
    user_message = json.dumps(memories, ensure_ascii=False)

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_message},
                    ],
                    temperature=TEMPERATURE,
                )
                content = response.choices[0].message.content or ""
                raw = extract_json_from_response(content)
                return _validate_partition(memories, raw)

            except Exception as exc:
                if attempt == MAX_RETRIES - 1:
                    print(f"  [WARN] Giving up after {MAX_RETRIES} attempts: {exc}")
                    fallback = {cat: [] for cat in CATEGORIES}
                    fallback["personal"] = list(memories)
                    return fallback
                await asyncio.sleep(2**attempt)

    # unreachable, but satisfies type checker
    fallback = {cat: [] for cat in CATEGORIES}
    fallback["personal"] = list(memories)
    return fallback


async def _process_sample(
    sample,
    client,
    semaphore: asyncio.Semaphore,
    out_file,
    lock: asyncio.Lock,
    counter: list[int],
    total: int,
) -> None:
    memories: list[str] = sample.memories

    if memories:
        partition = await _classify(client, memories, semaphore)
    else:
        partition = {cat: [] for cat in CATEGORIES}

    result: dict = {
        "query": sample.prompt,
        "memories": partition,
        "hash_id": sample.sample_id,
        "failure_type": "cim",
        "required_attributes": sample.required_attributes,
        "forbidden_attributes": sample.forbidden_attributes,
        "cim_metadata": sample.metadata,
    }
    # Also hoist top-level convenience fields that the benchmark runner expects
    if "cim_task" in sample.metadata:
        result["cim_task"] = sample.metadata["cim_task"]
    if "cim_recipient" in sample.metadata:
        result["cim_recipient"] = sample.metadata["cim_recipient"]

    async with lock:
        out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
        out_file.flush()
        counter[0] += 1
        print(f"[{counter[0]}/{total}] {sample.prompt[:70]}…")


async def main() -> None:
    labels_file = CIM_LABELS_FILE if CIM_LABELS_FILE.exists() else None
    if labels_file is None:
        print(
            "[WARN] CIM labels file not found at "
            f"{CIM_LABELS_FILE}. Falling back to HuggingFace label column."
        )

    print(f"Loading CIM dataset from {CIM_DATASET_ID} …")
    cim_dataset = CIMDataset(
        dataset_id=CIM_DATASET_ID,
        memory_mode="full_profile",
        labels_file=labels_file,
    )
    samples = list(cim_dataset)
    total = len(samples)
    print(f"Loaded {total} CIM samples")

    # Resume support
    done_ids = _load_checkpoint()
    pending = [s for s in samples if s.sample_id not in done_ids]
    print(f"Already done: {len(done_ids)} | Remaining: {len(pending)}")

    if not pending:
        print("All samples already processed.")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    counter = [len(done_ids)]

    async with get_vertex_ai_client(MODEL_LOCATION) as client:
        with open(OUTPUT_FILE, "a", encoding="utf-8") as out_file:
            tasks = [
                _process_sample(
                    sample, client, semaphore, out_file, lock, counter, total
                )
                for sample in pending
            ]
            await asyncio.gather(*tasks)

    print(f"\nDone! Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
