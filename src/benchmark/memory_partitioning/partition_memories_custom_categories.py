#!/usr/bin/env python3
"""Partition the 'memories' field in full_benchmark.jsonl into categories.

Each sample's flat list of memories is replaced with a dict of category keys.
The model sorts memories into the 11 predefined categories, and may also create
up to 2 new custom categories per sample when memories genuinely don't fit any
predefined category. All other sample fields are preserved exactly as-is.

Output is written incrementally so the script is safe to interrupt and resume.

─── HOW TO EDIT ───────────────────────────────────────────────────────────────
  • Change the model / location / temperature  →  MODEL block below
  • Change input/output paths                  →  RUN CONFIG below
  • Change the dataset you want to partition   →  RUN CONFIG below
  • Change concurrency or retry behaviour      →  RUN_CONFIG below
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
CONCURRENCY  = 7     # max simultaneous API requests
BATCH_MODE   = False # True = submit all requests via /v1/batches (cheaper, async, ~50% cost)
BATCH_POLL_INTERVAL = 15  # seconds between batch status polls (grows up to 60s)
MAX_RETRIES  = 5     # retry attempts per sample on parse / API failure
_PROJECT_ROOT = Path(__file__).parent.parent.parent
INPUT_FILE   = _PROJECT_ROOT / "benchmark_samples/cross_domain_and_beneficial.jsonl"
OUTPUT_FILE  = _PROJECT_ROOT / "benchmark_samples/partitioned_custom_categories/llama3p3_70b/full_benchmark.jsonl"
# ──────────────────────────────────────────────────────────────────────────────

# ── PROMPT ────────────────────────────────────────────────────────────────────
# The 11 fixed categories that are always present in output (even if empty).
DEFAULT_CATEGORIES = [
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

# For backward-compat; code that referenced CATEGORIES still works.
CATEGORIES = DEFAULT_CATEGORIES

# Constraints on model-defined custom category names.
_CUSTOM_CAT_MAX_NEW   = 2    # max new categories per sample
_CUSTOM_CAT_MIN_LEN   = 3
_CUSTOM_CAT_MAX_LEN   = 12

SYSTEM_PROMPT = """\
You are a memory classifier. Your task is to sort a list of personal memories
into exactly one category each.

Predefined categories — read the descriptions carefully before classifying:

personal     – the broadest catch-all for individual life outside structured domains:
              hobbies, sports, games, cooking, travel, leisure,
              entertainment, arts & crafts, music, reading, fashion, technology
              interests, outdoor activities, pets, gardening, philosophy, personal
              reflections, lifestyle choices, personality traits, values, opinions, 
              volunteering, and any other interest or pastime.
health       – physical or mental health, medical conditions, treatments,
              medications, fitness goals, therapy, disabilities, diet for health.
identity     – core self-concept: nationality, ethnicity, religion, spiritual
              practice, gender identity, sexuality, political ideology, deeply
              held beliefs that define who the person is.
social       – relationships and interactions with any other person who is NOT
              a romantic partner: family (parents, siblings, children, extended
              family), friends, neighbours, acquaintances, colleagues (socially).
romantic     – intimate or romantic relationships: dating, partners, marriage,
              attraction, breakups, divorce, jealousy, affection.
education    – schooling, degrees, courses, academic history, tutoring, exams,
              certifications, formal or informal learning experiences.
employment   – jobs, work history, career, workplace dynamics, colleagues
              (professionally), professional skills, freelance/business ventures.
finance      – money, savings, income, expenses, debt, investments, banking, taxes,
              insurance, financial goals.
housing      – home, residence, living situation, roommates, neighbours (housing),
              rent, mortgage, moving, home maintenance.
legal        – legal issues, contracts, court matters, rights, criminal record,
              official government documents, immigration status.
schedule     – appointments, routines, recurring events, time-based plans,
              deadlines, daily habits, reminders.

When to create a custom category:
A custom category is justified ONLY when ALL of the following are true:
 1. Multiple memories in this batch form a coherent, substantial life domain.
 2. That domain is genuinely absent from every predefined category above, or any new category already created.
 3. The domain cannot reasonably be called a sub-topic of one of the default or newly introduced categories.

Do NOT create custom categories for: lifestyle, leisure, entertainment,
sports, cooking, fashion, technology, gardening, garden, transport,
transportation, vehicles, arts, music, philosophy, history, language, community,
activism, environment, research, productivity, creative_work, interest, pastime, preference,
spiritual_practice, volunteer, volunteering, or any near-synonym of an existing category.
All of these belong in another predefined category.

If you do create a custom category:
 • Lowercase letters and underscores only, 3–15 characters.
 • Choose a single canonical name — do NOT create variants of the same concept
   (e.g. pick "travel" not both "travel" and "trips", or "allirgies" and "diet").
 • Create at most 2 custom categories per response.
 • Only include the key if it has at least one memory in it.

Rules:
1. Each memory must appear in exactly one category.
2. Do not drop or duplicate memories.
3. If a memory fits multiple categories, choose the most specific predefined one.
4. All 11 predefined keys must always be present (use [] if empty).
5. Custom category keys appear after the predefined ones.
6. Do not modify the memory text.

Return ONLY a single-line JSON object. Example with one justified custom category:

{"health": [...], "identity": [...], "social": [...], "romantic": [...], "personal": [...], "education": [...], "employment": [...], "finance": [...], "housing": [...], "legal": [...], "schedule": [...], "travel": [...]}

Omit any custom-category key if it has no memories.
"""

# ──────────────────────────────────────────────────────────────────────────────


# ── Internals (no need to edit below) ─────────────────────────────────────────

import re    # noqa: E402
import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent.parent))  # add src/ so 'benchmark' is importable

from benchmark.utils import extract_json_from_response, generate_hash_id, get_vertex_ai_client  # noqa: E402

_CUSTOM_CAT_RE = re.compile(r'^[a-z][a-z_]{1,}[a-z]$')

PARTITION_MAP = {
    "cross_domain":            "cross_domain.jsonl",
    "beneficial_memory_usage": "beneficial_samples.jsonl",
}


def _write_partitions() -> None:
    """Read the completed OUTPUT_FILE and split it into 3 files by failure_type."""
    out_dir = OUTPUT_FILE.parent
    partitions: dict[str, list[str]] = {key: [] for key in PARTITION_MAP}

    with open(OUTPUT_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            ft = sample.get("failure_type", "")
            if ft in partitions:
                partitions[ft].append(line)

    for failure_type, filename in PARTITION_MAP.items():
        out_path = out_dir / filename
        lines = partitions[failure_type]
        with open(out_path, "w") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")
        print(f"Wrote {len(lines):>3} samples [{failure_type}] → {out_path}")


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


def _is_valid_custom_category(name: str) -> bool:
    """Return True if name is an acceptable model-created category label."""
    return (
        name not in DEFAULT_CATEGORIES
        and _CUSTOM_CAT_MIN_LEN <= len(name) <= _CUSTOM_CAT_MAX_LEN
        and _CUSTOM_CAT_RE.match(name) is not None
    )


def _canonicalize_custom_name(name: str) -> str:
    """Normalize morphological variants to a canonical root for deduplication.

    Strips common English suffixes so that, e.g., "volunteering" and
    "volunteer" both resolve to "volunteer" and are merged into one key.
    Only strips when the remaining root is at least 4 characters.
    """
    if name.endswith("ing"):
        root = name[:-3]
        if len(root) >= 4:
            return root
    if name.endswith("tion") and len(name) > 7:
        return name[:-4]
    if name.endswith("s") and len(name) > 5:
        return name[:-1]
    return name


def _validate_partition(
    memories: list[str], raw: dict
) -> dict[str, list[str]]:
    """Ensure every input memory appears exactly once in the result.

    Processes all DEFAULT_CATEGORIES first, then accepts up to
    _CUSTOM_CAT_BLOCKLIST_CUSTOM_CAT_MAX_NEW model-created categories for memories that
    genuinely don't fit the predefined set. Any remaining unplaced
    memories fall back to 'personal'.
    """
    result: dict[str, list[str]] = {cat: [] for cat in DEFAULT_CATEGORIES}
    placed: set[str] = set()

    # Process predefined categories first.
    for cat in DEFAULT_CATEGORIES:
        for mem in raw.get(cat, []):
            if mem in memories and mem not in placed:
                result[cat].append(mem)
                placed.add(mem)

    # Accept custom categories proposed by the model (up to the limit).
    # Canonicalize names so morphological variants (e.g. "volunteering" vs
    # "volunteer") are merged into the same key.
    custom_accepted = 0
    seen_roots: dict[str, str] = {}  # canonical_root → accepted key name
    for cat, items in raw.items():
        if custom_accepted >= _CUSTOM_CAT_MAX_NEW:
            break
        if not _is_valid_custom_category(cat):
            continue
        root = _canonicalize_custom_name(cat)
        valid_items = [m for m in items if m in memories and m not in placed]
        if not valid_items:
            continue
        if root in seen_roots:
            # Merge into the already-accepted canonical key
            canonical = seen_roots[root]
            result[canonical].extend(valid_items)
        else:
            result[cat] = valid_items
            seen_roots[root] = cat
            custom_accepted += 1
        for mem in valid_items:
            placed.add(mem)

    # Fallback: anything the model missed goes to 'personal'.
    for mem in memories:
        if mem not in placed:
            result["personal"].append(mem)

    return result


async def _classify(
    client,
    memories: list[str],
    semaphore: asyncio.Semaphore,
) -> dict[str, list[str]]:
    """Call the LLM and return a validated partition (11 default + optional custom categories)."""
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

    # Compute hash from the original flat memories BEFORE partitioning.
    # This stable hash matches what the benchmark runner computes for the same
    # sample from the unpartitioned full_benchmark.jsonl, so all models share
    # one canonical hash_id regardless of how their memories are categorised.
    hash_id = sample.get("hash_id") or generate_hash_id(memories, sample["query"])

    if memories:
        partition = await _classify(client, memories, semaphore)
    else:
        partition = {cat: [] for cat in CATEGORIES}

    # Preserve every original field; replace 'memories' with partition and pin hash_id.
    result = {**sample, "hash_id": hash_id, "memories": partition}

    async with lock:
        out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
        out_file.flush()
        counter[0] += 1
        print(f"[{counter[0]}/{total}] {sample['query'][:70]}…")


async def _run_batch(pending: list[dict], client) -> None:
    """Submit all pending samples as one batch job via the OpenAI /v1/batches API.

    Each sample becomes one line in a JSONL request file.  After the job
    completes the output file is downloaded, parsed, validated, and written
    to OUTPUT_FILE incrementally — identical post-processing to the online path.
    """
    import tempfile

    # ── Build id → sample mapping (hash_id is the custom_id key) ──────────────
    id_to_sample: dict[str, dict] = {}
    batch_lines: list[str] = []

    for sample in pending:
        memories: list[str] = sample.get("memories", [])
        hash_id = sample.get("hash_id") or generate_hash_id(memories, sample["query"])
        id_to_sample[hash_id] = {**sample, "_computed_hash_id": hash_id}

        request_line = {
            "custom_id": hash_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": json.dumps(memories, ensure_ascii=False)},
                ],
                "temperature": TEMPERATURE,
            },
        }
        batch_lines.append(json.dumps(request_line, ensure_ascii=False))

    # ── Write requests to a temp file and upload ───────────────────────────────
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
        tmp.write("\n".join(batch_lines))
        tmp_path = Path(tmp.name)

    try:
        print(f"Uploading {len(pending)} requests…")
        with open(tmp_path, "rb") as f:
            uploaded = await client.files.create(file=f, purpose="batch")

        print(f"Submitting batch job (input_file_id={uploaded.id})…")
        batch = await client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        print(f"Batch job created: {batch.id}")

        # ── Poll until terminal status ─────────────────────────────────────────
        poll_interval = BATCH_POLL_INTERVAL
        terminal = {"completed", "failed", "expired", "cancelled"}
        while batch.status not in terminal:
            await asyncio.sleep(poll_interval)
            batch = await client.batches.retrieve(batch.id)
            c = batch.request_counts
            print(
                f"  [{batch.status}] completed={c.completed} "
                f"failed={c.failed} total={c.total}"
            )
            poll_interval = min(poll_interval * 1.5, 60)

        if batch.status != "completed":
            raise RuntimeError(f"Batch job ended with status '{batch.status}' (id={batch.id})")

        # ── Download results ───────────────────────────────────────────────────
        print(f"Batch complete. Downloading results (output_file_id={batch.output_file_id})…")
        result_bytes = await client.files.content(batch.output_file_id)
        result_lines = result_bytes.text.splitlines()

        # Clean up uploaded input file (best-effort)
        try:
            await client.files.delete(uploaded.id)
        except Exception:
            pass

    finally:
        tmp_path.unlink(missing_ok=True)

    # ── Parse and write results ────────────────────────────────────────────────
    processed = 0
    with open(OUTPUT_FILE, "a") as out_file:
        for line in result_lines:
            if not line.strip():
                continue
            result_obj = json.loads(line)
            custom_id = result_obj.get("custom_id", "")
            sample = id_to_sample.get(custom_id)
            if sample is None:
                print(f"  [WARN] Unknown custom_id in batch output: {custom_id}")
                continue

            memories = sample.get("memories", [])
            hash_id  = sample["_computed_hash_id"]
            clean_sample = {k: v for k, v in sample.items() if k != "_computed_hash_id"}

            error        = result_obj.get("error")
            response_body = (result_obj.get("response") or {}).get("body") or {}

            if error or not response_body:
                print(f"  [WARN] Batch error for {custom_id[:16]}…: {error}")
                partition = {cat: [] for cat in CATEGORIES}
                partition["personal"] = list(memories)
            else:
                choices = response_body.get("choices", [])
                content = choices[0]["message"]["content"] if choices else ""
                try:
                    raw = extract_json_from_response(content)
                    partition = _validate_partition(memories, raw)
                except Exception as exc:
                    print(f"  [WARN] Parse error for {custom_id[:16]}…: {exc}")
                    partition = {cat: [] for cat in CATEGORIES}
                    partition["personal"] = list(memories)

            output = {**clean_sample, "hash_id": hash_id, "memories": partition}
            out_file.write(json.dumps(output, ensure_ascii=False) + "\n")
            out_file.flush()
            processed += 1
            print(f"[{processed}/{len(pending)}] {clean_sample['query'][:70]}…")

    print(f"Batch: processed {processed}/{len(pending)} results.")


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
        print("All samples already processed. Writing partitions from existing output…")
        _write_partitions()
        return

    if BATCH_MODE:
        print("Running in batch mode (OpenAI /v1/batches API)…")
        async with get_vertex_ai_client(MODEL_LOCATION) as client:
            await _run_batch(pending, client)
    else:
        semaphore = asyncio.Semaphore(CONCURRENCY)
        lock = asyncio.Lock()
        counter = [len(done_queries)]   # mutable int shared across coroutines

        print(f"Running in concurrent mode (max {CONCURRENCY})…")

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
    print("Writing partition files…")
    _write_partitions()


if __name__ == "__main__":
    asyncio.run(main())
