#!/usr/bin/env python3
"""Partition the 'memories' field using cosine similarity to 11 fixed categories.

Each sample's flat list of memories is replaced with a dict of 11 category keys,
with all other sample fields (query, memory_domain, query_domain, failure_type, …)
preserved exactly as-is.

Uses sentence-transformers + sklearn cosine similarity — no LLM calls required.

─── HOW TO EDIT ───────────────────────────────────────────────────────────────
  • Change the embedding model          →  MODEL_NAME below
  • Change input/output paths           →  RUN CONFIG below
  • Change the domain descriptions      →  DOMAINS below
───────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── MODEL ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "BAAI/bge-large-en-v1.5"
# ──────────────────────────────────────────────────────────────────────────────

# ── RUN CONFIG ────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent.parent
INPUT_FILE  = _PROJECT_ROOT / "benchmark_samples/persistbench/baseline/cross_domain_and_beneficial.jsonl"
OUTPUT_FILE = _PROJECT_ROOT / "benchmark_samples/persistbench/partitioned/cos_similarity/full_benchmark.jsonl"
# ──────────────────────────────────────────────────────────────────────────────

# ── CATEGORIES ────────────────────────────────────────────────────────────────
CATEGORIES = [
    "health",
    "identity",
    "social",
    "romantic",
    "personal",
    "education",
    "employment",
    "finance",
    "housing",
    "legal",
    "schedule",
]

# One descriptive sentence per category — used as the embedding target.
DOMAINS = [
    "health – physical or mental health, medical conditions, treatments, medications, fitness, therapy",
    "identity – core personal identity traits such as nationality, religion, gender identity, values, beliefs",
    "social – non-romantic relationships and interactions with friends, family, acquaintances, or colleagues",
    "romantic – intimate or romantic relationships including dating, partners, marriage, attraction, breakups",
    "personal – hobbies, preferences, lifestyle choices, personality traits, interests",
    "education – schooling, degrees, courses, academic history, tutoring, learning experiences",
    "employment – jobs, work history, workplace experiences, colleagues, professional skills",
    "finance – money, savings, income, expenses, debt, investments, banking, taxes",
    "housing – home, residence, living situation, roommates, neighbors, rent, mortgage",
    "legal – legal issues, contracts, court matters, rights, criminal record, official documents",
    "schedule – appointments, routines, recurring events, time-based plans, daily habits",
]

assert len(CATEGORIES) == len(DOMAINS) == 11
# ──────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmark.utils import generate_hash_id  # noqa: E402


def _load_model() -> tuple[SentenceTransformer, np.ndarray]:
    print(f"Loading embedding model '{MODEL_NAME}'…")
    model = SentenceTransformer(MODEL_NAME)
    domain_embeddings = model.encode(DOMAINS, show_progress_bar=False)
    return model, domain_embeddings


def _partition_memories(
    memories: list[str],
    model: SentenceTransformer,
    domain_embeddings: np.ndarray,
) -> dict[str, list[str]]:
    """Assign each memory to the category with the highest cosine similarity."""
    result: dict[str, list[str]] = {cat: [] for cat in CATEGORIES}
    if not memories:
        return result

    memory_embeddings = model.encode(memories, show_progress_bar=False)
    scores = cosine_similarity(memory_embeddings, domain_embeddings)  # (N, 11)

    for i, memory in enumerate(memories):
        best_cat = CATEGORIES[int(np.argmax(scores[i]))]
        result[best_cat].append(memory)

    return result


def _write_partitions(output_file: Path) -> None:
    """Split the completed output file into per-failure_type files."""
    partition_map = {
        "cross_domain":            "cross_domain.jsonl",
        "sycophancy":              "sycophancy.jsonl",
        "beneficial_memory_usage": "beneficial_samples.jsonl",
    }
    out_dir = output_file.parent
    buckets: dict[str, list[str]] = {k: [] for k in partition_map}

    with open(output_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            ft = sample.get("failure_type", "")
            if ft in buckets:
                buckets[ft].append(line)

    for failure_type, filename in partition_map.items():
        out_path = out_dir / filename
        lines = buckets[failure_type]
        with open(out_path, "w") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")
        print(f"Wrote {len(lines):>3} samples [{failure_type}] → {out_path}")


def main() -> None:
    # Load input samples
    samples: list[dict] = []
    with open(INPUT_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"Loaded {len(samples)} samples from {INPUT_FILE}")

    # Resume support — skip already-processed queries
    done_queries: set[str] = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done_queries.add(json.loads(line)["query"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    pending = [s for s in samples if s["query"] not in done_queries]
    print(f"Already done: {len(done_queries)} | Remaining: {len(pending)}")

    if not pending:
        print("All samples already processed. Writing partition files from existing output…")
        _write_partitions(OUTPUT_FILE)
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    model, domain_embeddings = _load_model()

    total = len(samples)
    counter = len(done_queries)

    with open(OUTPUT_FILE, "a") as out_file:
        for sample in pending:
            memories: list[str] = sample.get("memories", [])
            hash_id = sample.get("hash_id") or generate_hash_id(memories, sample["query"])
            partition = _partition_memories(memories, model, domain_embeddings)

            result = {**sample, "hash_id": hash_id, "memories": partition}
            out_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_file.flush()

            counter += 1
            print(f"[{counter}/{total}] {sample['query'][:70]}…")

    print(f"\nDone! Saved to {OUTPUT_FILE}")
    print("Writing partition files…")
    _write_partitions(OUTPUT_FILE)


if __name__ == "__main__":
    main()
