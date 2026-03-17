#!/usr/bin/env python3
"""Compare partitioned memory files across models.

Discovers all model directories under benchmark_samples/partitioned/ automatically.
For each pair of models (and across all models), reports:
  - How many entries each model holds per category
  - Which categories are withheld per entry (one model has content, another has none)
  - Memory content overlap (exact shared items vs. model-exclusive items)
  - Per-domain breakdown of withheld categories

Usage:
    python analysis/compare_partitions.py                         # auto-discover all models
    python analysis/compare_partitions.py --file full_benchmark   # specific file (no .jsonl)
    python analysis/compare_partitions.py --models llama3p3_70b qwen3_235b
    python analysis/compare_partitions.py --verbose               # show example diffs
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict
from itertools import combinations

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PARTITIONED_DIR = REPO_ROOT / "benchmark_samples" / "partitioned"

CATEGORIES = [
    "personal", "education", "employment", "finance", "housing",
    "legal", "health", "schedule", "identity", "social", "romantic",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def discover_models(file_stem: str) -> list[tuple[str, pathlib.Path]]:
    """Return [(model_name, path)] for every model dir that has <file_stem>.jsonl."""
    results = []
    for model_dir in sorted(PARTITIONED_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        candidate = model_dir / f"{file_stem}.jsonl"
        if candidate.exists():
            results.append((model_dir.name, candidate))
    return results


def load_entries(path: pathlib.Path) -> dict[str, dict]:
    """Load a .jsonl file into {hash_id: entry}."""
    entries = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                e = json.loads(line)
                entries[e["hash_id"]] = e
    return entries


# ---------------------------------------------------------------------------
# Per-model stats
# ---------------------------------------------------------------------------

def category_fill_stats(entries: dict[str, dict]) -> dict[str, dict]:
    """For each category: total non-empty entries and total memory items."""
    stats: dict[str, dict] = {c: {"entries_with_content": 0, "total_items": 0} for c in CATEGORIES}
    for entry in entries.values():
        mems = entry.get("memories", {})
        if not isinstance(mems, dict):
            continue
        for cat in CATEGORIES:
            items = mems.get(cat) or []
            if items:
                stats[cat]["entries_with_content"] += 1
                stats[cat]["total_items"] += len(items)
    return stats


# ---------------------------------------------------------------------------
# Pairwise comparison
# ---------------------------------------------------------------------------

def pairwise_diff(
    entries_a: dict[str, dict],
    entries_b: dict[str, dict],
) -> dict:
    """Compare two models' entries over their shared hash_ids."""
    shared_ids = set(entries_a) & set(entries_b)

    cat_stats: dict[str, dict[str, int]] = {
        c: {"a_only": 0, "b_only": 0, "both": 0, "neither": 0}
        for c in CATEGORIES
    }

    content_diffs = 0  # entries where any category has different items (ignoring order)
    withheld_diffs = 0  # entries where at least one category is withheld from one model

    examples: list[dict] = []  # up to 3 interesting examples

    for hid in shared_ids:
        mems_a = entries_a[hid].get("memories", {})
        mems_b = entries_b[hid].get("memories", {})
        if not isinstance(mems_a, dict) or not isinstance(mems_b, dict):
            continue

        entry_has_content_diff = False
        entry_has_withheld = False
        diff_cats = []

        for cat in CATEGORIES:
            items_a = set(mems_a.get(cat) or [])
            items_b = set(mems_b.get(cat) or [])

            has_a = bool(items_a)
            has_b = bool(items_b)

            if has_a and has_b:
                cat_stats[cat]["both"] += 1
            elif has_a:
                cat_stats[cat]["a_only"] += 1
                entry_has_withheld = True
            elif has_b:
                cat_stats[cat]["b_only"] += 1
                entry_has_withheld = True
            else:
                cat_stats[cat]["neither"] += 1

            if sorted(items_a) != sorted(items_b):
                entry_has_content_diff = True
                diff_cats.append(cat)

        if entry_has_content_diff:
            content_diffs += 1
        if entry_has_withheld:
            withheld_diffs += 1

        if len(examples) < 3 and entry_has_withheld and diff_cats:
            examples.append({
                "hash_id": hid,
                "diff_categories": diff_cats,
                "memory_domain": entries_a[hid].get("memory_domain", ""),
                "failure_type": entries_a[hid].get("failure_type", ""),
            })

    return {
        "shared_entries": len(shared_ids),
        "content_diffs": content_diffs,
        "withheld_diffs": withheld_diffs,
        "category_stats": cat_stats,
        "examples": examples,
    }


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _bar(n: int, total: int, width: int = 20) -> str:
    if total == 0:
        return " " * width
    filled = round(n / total * width)
    return "#" * filled + "." * (width - filled)


def print_model_summary(model: str, entries: dict) -> None:
    stats = category_fill_stats(entries)
    n = len(entries)
    print(f"\n  {model}  ({n} entries)")
    print(f"  {'Category':<14}  {'Filled':>7}  {'%':>5}  {'Items':>6}  bar")
    print(f"  {'-'*14}  {'-'*7}  {'-'*5}  {'-'*6}  {'-'*20}")
    for cat in CATEGORIES:
        s = stats[cat]
        filled = s["entries_with_content"]
        pct = filled / n * 100 if n else 0
        bar = _bar(filled, n)
        print(f"  {cat:<14}  {filled:>7}  {pct:>4.1f}%  {s['total_items']:>6}  {bar}")


def print_pairwise(name_a: str, name_b: str, diff: dict, verbose: bool) -> None:
    n = diff["shared_entries"]
    print(f"\n  {'Category':<14}  {'A-only':>7}  {'B-only':>7}  {'Both':>7}  {'Neither':>8}")
    print(f"  {'-'*14}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}")
    for cat in CATEGORIES:
        s = diff["category_stats"][cat]
        print(
            f"  {cat:<14}  {s['a_only']:>7}  {s['b_only']:>7}  {s['both']:>7}  {s['neither']:>8}"
        )
    print()
    if n:
        print(f"  Shared entries:              {n}")
        print(f"  Entries with content diffs:  {diff['content_diffs']:>4}  ({diff['content_diffs']/n*100:.1f}%)")
        print(f"  Entries with withheld cats:  {diff['withheld_diffs']:>4}  ({diff['withheld_diffs']/n*100:.1f}%)")

    if verbose and diff["examples"]:
        print(f"\n  Example differing entries:")
        for ex in diff["examples"]:
            print(f"    {ex['hash_id'][:20]}  failure={ex['failure_type']:<15}  domain={ex['memory_domain']}")
            print(f"      differing categories: {', '.join(ex['diff_categories'])}")


def print_coverage_matrix(model_names: list[str], all_entries: dict[str, dict[str, dict]]) -> None:
    """For each category, show what % of entries each model covers."""
    col_w = max(len(m) for m in model_names) + 2
    print(f"\n  Coverage matrix  (% entries with content per category)")
    header = f"  {'Category':<14}" + "".join(f"  {m[:col_w]:<{col_w}}" for m in model_names)
    print(header)
    print("  " + "-" * (14 + (col_w + 2) * len(model_names)))

    for cat in CATEGORIES:
        row = f"  {cat:<14}"
        for m in model_names:
            entries = all_entries[m]
            covered = sum(
                1 for e in entries.values()
                if bool((e.get("memories") or {}).get(cat))
            )
            pct = covered / len(entries) * 100 if entries else 0
            row += f"  {pct:>{col_w-1}.1f}%"
        print(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare partitioned memories across models")
    parser.add_argument(
        "--file", default="full_benchmark",
        help="File stem to compare (default: full_benchmark)"
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Restrict to these model dir names (default: auto-discover all)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print example differing entries for each pair"
    )
    args = parser.parse_args()

    available = discover_models(args.file)
    if not available:
        print(f"No models found with '{args.file}.jsonl' under {PARTITIONED_DIR}")
        return

    if args.models:
        available = [(n, p) for n, p in available if n in args.models]
        if not available:
            print(f"None of the requested models found. Available: {[n for n, _ in discover_models(args.file)]}")
            return

    print(f"\nFile: {args.file}.jsonl")
    print(f"Models: {[n for n, _ in available]}")

    all_entries: dict[str, dict[str, dict]] = {
        model_name: load_entries(path)
        for model_name, path in available
    }
    model_names = [n for n, _ in available]

    # ── Per-model summaries ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("PER-MODEL CATEGORY FILL")
    print(f"{'='*70}")
    for model_name in model_names:
        print_model_summary(model_name, all_entries[model_name])

    # ── Coverage matrix ──────────────────────────────────────────────────────
    if len(model_names) > 1:
        print(f"\n{'='*70}")
        print("COVERAGE MATRIX")
        print(f"{'='*70}")
        print_coverage_matrix(model_names, all_entries)

    # ── Pairwise comparisons ─────────────────────────────────────────────────
    if len(model_names) > 1:
        print(f"\n{'='*70}")
        print("PAIRWISE COMPARISONS")
        print(f"{'='*70}")
        for a, b in combinations(model_names, 2):
            print(f"\n  A = {a}")
            print(f"  B = {b}")
            diff = pairwise_diff(all_entries[a], all_entries[b])
            print_pairwise(a, b, diff, args.verbose)

    print()


if __name__ == "__main__":
    main()
