"""Compare two checkpoint files judged by different models and produce a disagreement analysis.

Usage:
    uv run python src/benchmark/judge_disagreement_analysis.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Add project root to sys.path so we can import from the benchmark package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.metrics_cim import _should_filter_private_attr  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_checkpoint(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_revealed_set_and_evidence(judge: dict, forbidden_attrs: set[str]) -> tuple[set[str], dict[str, str]]:
    """Return (set of forbidden attrs judged as revealed, {attr: explanation}).

    Handles both judge output formats:
      - Old format: revealed_attributes is dict of attr->0/1,
                    revealed_attributes_explanation is dict of attr->str
      - New format: revealed_attributes is list of revealed attr names,
                    per_attribute is list of {id, revealed, used, evidence}
    """
    ra = judge.get("revealed_attributes", {})
    revealed: set[str] = set()
    evidence: dict[str, str] = {}

    if isinstance(ra, dict):
        # Old format
        for attr, val in ra.items():
            if attr in forbidden_attrs and int(val) == 1:
                revealed.add(attr)
        explanations = judge.get("revealed_attributes_explanation", {})
        if isinstance(explanations, dict):
            for attr in forbidden_attrs:
                if attr in explanations:
                    evidence[attr] = str(explanations[attr])
    elif isinstance(ra, list):
        # New format – revealed_attributes is just the list of revealed names
        revealed = set(ra) & forbidden_attrs
        per_attr = judge.get("per_attribute", [])
        if isinstance(per_attr, list):
            for item in per_attr:
                attr_id = item.get("id", "")
                if attr_id in forbidden_attrs:
                    ev = item.get("evidence", "")
                    if ev:
                        evidence[attr_id] = str(ev)

    return revealed, evidence


def _run_comparison(
    path_a: Path,
    path_b: Path,
    label_a: str = "Gemini",
    label_b: str = "Kimi",
    max_examples: int = 10,
) -> None:
    """Compare two checkpoint files and print disagreement analysis."""
    data_a = _load_checkpoint(path_a)
    data_b = _load_checkpoint(path_b)

    entries_a = data_a.get("entries", {})
    entries_b = data_b.get("entries", {})

    common_ids = set(entries_a.keys()) & set(entries_b.keys())
    if not common_ids:
        print("  No common entry IDs found – nothing to compare.\n")
        return

    # Counters
    total = 0
    categories: Counter = Counter()
    attr_disagreements: Counter = Counter()  # attr_name -> count
    examples: list[dict] = []

    for eid in sorted(common_ids):
        entry_a = entries_a[eid]
        entry_b = entries_b[eid]

        forbidden = set(entry_a.get("forbidden_attributes", []))
        # Filter private attrs the same way metrics_cim does
        filtered_forbidden = {a for a in forbidden if not _should_filter_private_attr(a)}
        if not filtered_forbidden:
            continue

        name = entry_a.get("cim_metadata", {}).get("name", "?")
        query = entry_a.get("query", "")

        # Iterate over models present in *both* checkpoints
        models_a = set(entry_a.get("results", {}).keys())
        models_b = set(entry_b.get("results", {}).keys())
        common_models = models_a & models_b

        for model_name in sorted(common_models):
            gens_a = entry_a["results"][model_name].get("generations", [])
            gens_b = entry_b["results"][model_name].get("generations", [])

            for gen_idx in range(min(len(gens_a), len(gens_b))):
                gen_a = gens_a[gen_idx]
                gen_b = gens_b[gen_idx]
                judge_a = gen_a.get("judge")
                judge_b = gen_b.get("judge")
                if not judge_a or not judge_b:
                    continue

                revealed_a, evidence_a = _get_revealed_set_and_evidence(judge_a, filtered_forbidden)
                revealed_b, evidence_b = _get_revealed_set_and_evidence(judge_b, filtered_forbidden)

                memory_response = gen_a.get("memory_response", "") or ""

                for attr in sorted(filtered_forbidden):
                    a_says = attr in revealed_a
                    b_says = attr in revealed_b
                    total += 1

                    if a_says and b_says:
                        cat = "Both agree violated"
                    elif not a_says and not b_says:
                        cat = "Both agree clean"
                    elif a_says and not b_says:
                        cat = f"{label_a}-only violation"
                    else:
                        cat = f"{label_b}-only violation"

                    categories[cat] += 1

                    if a_says != b_says:
                        attr_disagreements[attr] += 1
                        if len(examples) < max_examples:
                            examples.append({
                                "name": name,
                                "query": query[:100],
                                "attr": attr,
                                "memory_response": memory_response[:200],
                                "a_revealed": a_says,
                                "b_revealed": b_says,
                                "a_evidence": evidence_a.get(attr, "(none)"),
                                "b_evidence": evidence_b.get(attr, "(none)"),
                                "category": cat,
                            })

    # ---- Print summary ----
    print(f"  Total private-attribute judgments: {total}")
    if total == 0:
        print("  (nothing to report)\n")
        return

    for cat in [
        "Both agree clean",
        "Both agree violated",
        f"{label_a}-only violation",
        f"{label_b}-only violation",
    ]:
        cnt = categories.get(cat, 0)
        pct = 100.0 * cnt / total
        print(f"  {cat:30s}: {cnt:5d}  ({pct:5.1f}%)")

    disagree_total = (
        categories.get(f"{label_a}-only violation", 0)
        + categories.get(f"{label_b}-only violation", 0)
    )
    print(f"\n  Total disagreements: {disagree_total}  ({100.0 * disagree_total / total:.1f}%)")

    # Top disagreement attributes
    if attr_disagreements:
        print(f"\n  Top attributes with most disagreements:")
        for attr, cnt in attr_disagreements.most_common(15):
            print(f"    {attr:55s}  {cnt}")

    # Detailed examples
    if examples:
        print(f"\n  === Detailed disagreement examples (up to {max_examples}) ===")
        for i, ex in enumerate(examples, 1):
            print(f"\n  --- Example {i} [{ex['category']}] ---")
            print(f"  Person : {ex['name']}")
            print(f"  Query  : {ex['query']}...")
            print(f"  Attr   : {ex['attr']}")
            print(f"  Response: {ex['memory_response']}...")
            print(f"  {label_a} says revealed={ex['a_revealed']}")
            print(f"    evidence: {ex['a_evidence'][:200]}")
            print(f"  {label_b} says revealed={ex['b_revealed']}")
            print(f"    evidence: {ex['b_evidence'][:200]}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    output_dir = PROJECT_ROOT / "outputs" / "CIM"

    comparisons = [
        {
            "title": "BASELINE (paper replication)",
            "path_a": output_dir / "baseline" / "cim_paper_replication_gemini.json",
            "path_b": output_dir / "baseline" / "cim_paper_replication_kimi.json",
        },
        {
            "title": "PARTITIONED memories",
            "path_a": output_dir / "partitioned" / "cim_partitioned_gemini.json",
            "path_b": output_dir / "partitioned" / "cim_partitioned_kimi.json",
        },
    ]

    for comp in comparisons:
        print("=" * 70)
        print(f"  {comp['title']}")
        print(f"  File A (Gemini): {comp['path_a']}")
        print(f"  File B (Kimi)  : {comp['path_b']}")
        print("=" * 70)

        if not comp["path_a"].exists():
            print(f"  SKIPPED – {comp['path_a']} not found\n")
            continue
        if not comp["path_b"].exists():
            print(f"  SKIPPED – {comp['path_b']} not found\n")
            continue

        _run_comparison(comp["path_a"], comp["path_b"])


if __name__ == "__main__":
    main()
