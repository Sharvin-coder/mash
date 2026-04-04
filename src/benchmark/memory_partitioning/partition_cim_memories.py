#!/usr/bin/env python3
"""Partition CIM memories into 11 categories — once per persona.

Reads samples from the facebook/CIMemories HuggingFace dataset (full_profile
mode), partitions each *persona's* flat memory list into 11 categories using an
LLM (one LLM call per persona, not per sample), and writes a JSONL file
compatible with the benchmark runner's partitioned mode.

Each output row preserves all CIM-specific fields (required_attributes,
forbidden_attributes, cim_metadata) so the judge can evaluate properly.

Usage:
  # Partition specific personas only
  uv run python partition_cim_memories.py --personas "Jeffery Day" "Shawn Franklin"

  # Partition all personas
  uv run python partition_cim_memories.py

─── HOW TO EDIT ───────────────────────────────────────────────────────────────
  * Change the model / location / temperature  ->  MODEL block below
  * Change input/output paths                  ->  RUN CONFIG below
  * Change concurrency or retry behaviour      ->  RUN CONFIG below
  * Change what the LLM is told to do          ->  SYSTEM_PROMPT below
───────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Partition CIM memories into 11 categories (once per persona).",
    )
    parser.add_argument(
        "--personas",
        nargs="+",
        default=None,
        help='Persona names to process (e.g. --personas "Jeffery Day" "Shawn Franklin"). '
             "Omit to process all personas.",
    )
    return parser.parse_args()


def _load_checkpoint() -> tuple[set[str], dict[str, dict[str, list[str]]]]:
    """Return (done_hash_ids, persona_partitions) from the existing output file.

    In addition to tracking which sample hash_ids are already written, this
    recovers the partition for each persona so that a resumed run does not need
    to re-call the LLM for personas that were already (partially) written.
    """
    done: set[str] = set()
    persona_partitions: dict[str, dict[str, list[str]]] = {}

    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    done.add(row["hash_id"])
                    # Recover the partition for this persona (first row wins)
                    name = row.get("cim_metadata", {}).get("name")
                    if name and name not in persona_partitions:
                        persona_partitions[name] = row["memories"]
                except (json.JSONDecodeError, KeyError):
                    pass

    return done, persona_partitions


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


def _build_sample_row(sample, partition: dict[str, list[str]]) -> dict:
    """Build the output JSONL row for a single sample."""
    result: dict = {
        "query": sample.prompt,
        "memories": partition,
        "hash_id": sample.sample_id,
        "failure_type": "cim",
        "required_attributes": sample.required_attributes,
        "forbidden_attributes": sample.forbidden_attributes,
        "cim_metadata": sample.metadata,
    }
    # Hoist top-level convenience fields that the benchmark runner expects
    if "cim_task" in sample.metadata:
        result["cim_task"] = sample.metadata["cim_task"]
    if "cim_recipient" in sample.metadata:
        result["cim_recipient"] = sample.metadata["cim_recipient"]
    return result


async def main() -> None:
    args = _parse_args()

    labels_file = CIM_LABELS_FILE if CIM_LABELS_FILE.exists() else None
    if labels_file is None:
        print(
            "[WARN] CIM labels file not found at "
            f"{CIM_LABELS_FILE}. Falling back to HuggingFace label column."
        )

    print(f"Loading CIM dataset from {CIM_DATASET_ID} ...")
    cim_dataset = CIMDataset(
        dataset_id=CIM_DATASET_ID,
        memory_mode="full_profile",
        labels_file=labels_file,
    )
    samples = list(cim_dataset)
    print(f"Loaded {len(samples)} CIM samples")

    # ── Group samples by persona ─────────────────────────────────────────────
    persona_samples: dict[str, list] = defaultdict(list)
    for s in samples:
        persona_samples[s.metadata["name"]].append(s)

    # Apply persona filter from CLI
    if args.personas is not None:
        requested = set(args.personas)
        available = set(persona_samples.keys())
        unknown = requested - available
        if unknown:
            print(f"[WARN] Unknown persona(s): {unknown}")
            print(f"       Available: {sorted(available)}")
        persona_samples = {
            name: samps for name, samps in persona_samples.items()
            if name in requested
        }

    if not persona_samples:
        print("No personas to process.")
        return

    total_samples = sum(len(samps) for samps in persona_samples.values())
    print(
        f"Processing {len(persona_samples)} persona(s), "
        f"{total_samples} samples total"
    )

    # ── Resume support ───────────────────────────────────────────────────────
    done_ids, persona_partitions = _load_checkpoint()
    print(f"Checkpoint: {len(done_ids)} samples written, "
          f"{len(persona_partitions)} persona partition(s) recovered")

    # ── Phase 1: Classify once per persona ───────────────────────────────────
    personas_to_classify = [
        name for name in persona_samples
        if name not in persona_partitions
    ]

    if personas_to_classify:
        print(f"\nPhase 1: Classifying {len(personas_to_classify)} persona(s) ...")
        semaphore = asyncio.Semaphore(CONCURRENCY)

        async with get_vertex_ai_client(MODEL_LOCATION) as client:
            classify_tasks = []
            for name in personas_to_classify:
                # All samples for the same persona share identical memories
                # in full_profile mode, so we use the first sample's list.
                memories = persona_samples[name][0].memories
                classify_tasks.append(_classify(client, memories, semaphore))

            results = await asyncio.gather(*classify_tasks)

            for name, partition in zip(personas_to_classify, results):
                persona_partitions[name] = partition
                mem_count = sum(len(v) for v in partition.values())
                print(f"  {name}: {mem_count} memories classified")
    else:
        print("\nPhase 1: All persona partitions recovered from checkpoint.")

    # ── Phase 2: Write sample rows ───────────────────────────────────────────
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    counter = len(done_ids)

    print(f"\nPhase 2: Writing sample rows ...")
    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_file:
        for name, samps in persona_samples.items():
            partition = persona_partitions[name]
            for sample in samps:
                if sample.sample_id in done_ids:
                    continue
                row = _build_sample_row(sample, partition)
                out_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_file.flush()
                counter += 1
                print(f"[{counter}/{total_samples}] {name}: {sample.prompt[:60]}...")

    print(f"\nDone! {counter} rows saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
