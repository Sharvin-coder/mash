#!/usr/bin/env python3
"""Partition the 'memories' field in full_benchmark.jsonl into 9 categories.

Each sample's flat list of memories is replaced with a dict of 9 category keys,
with all other sample fields (query, memory_domain, query_domain, failure_type, …)
preserved exactly as-is.

Output is written incrementally so the script is safe to interrupt and resume.

─── HOW TO EDIT ───────────────────────────────────────────────────────────────
  • Change the model / location / temperature  →  MODEL block below
  • Change what the LLM is told to do          →  SYSTEM_PROMPT below
  • Change concurrency or retry behaviour      →  RUN_CONFIG below
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
MODEL_NAME     = "qwen/qwen3-235b-a22b-instruct-2507-maas"
MODEL_LOCATION = "global"   # VertexAI region where the model is deployed
TEMPERATURE    = 0
# ──────────────────────────────────────────────────────────────────────────────

# ── PROMPT ────────────────────────────────────────────────────────────────────
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

# ── RUN CONFIG ────────────────────────────────────────────────────────────────
CONCURRENCY  = 10    # max simultaneous API requests
MAX_RETRIES  = 3     # retry attempts per sample on parse / API failure
_PROJECT_ROOT = Path(__file__).parent.parent.parent
INPUT_FILE   = _PROJECT_ROOT / "benchmark_samples/beneficial_samples.jsonl"
OUTPUT_FILE  = _PROJECT_ROOT / "benchmark_samples/partitioned/beneficial_samples_qwen3_235b.jsonl"
# ──────────────────────────────────────────────────────────────────────────────


# ── Internals (no need to edit below) ─────────────────────────────────────────

from benchmark.utils import extract_json_from_response, get_vertex_ai_client  # noqa: E402


def _load_checkpoint() -> set[str]:
    """Return the set of queries already written to the output file."""
    done: set[str] = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["query"])
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
    """Call the LLM and return a validated 9-category partition."""
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
    sample: dict,
    client,
    semaphore: asyncio.Semaphore,
    out_file,
    lock: asyncio.Lock,
    counter: list[int],
    total: int,
) -> None:
    memories: list[str] = sample.get("memories", [])

    if memories:
        partition = await _classify(client, memories, semaphore)
    else:
        partition = {cat: [] for cat in CATEGORIES}

    # Preserve every original field; only replace 'memories'
    result = {**sample, "memories": partition}

    async with lock:
        out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
        out_file.flush()
        counter[0] += 1
        print(f"[{counter[0]}/{total}] {sample['query'][:70]}…")


async def main() -> None:
    # Load input
    samples: list[dict] = []
    with open(INPUT_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    total = len(samples)
    print(f"Loaded {total} samples from {INPUT_FILE}")

    # Resume support
    done_queries = _load_checkpoint()
    pending = [s for s in samples if s["query"] not in done_queries]
    print(f"Already done: {len(done_queries)} | Remaining: {len(pending)}")

    if not pending:
        print("All samples already processed. Nothing to do.")
        return

    semaphore = asyncio.Semaphore(CONCURRENCY)
    lock = asyncio.Lock()
    counter = [len(done_queries)]   # mutable int shared across coroutines

    async with get_vertex_ai_client(MODEL_LOCATION) as client:
        with open(OUTPUT_FILE, "a") as out_file:
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
