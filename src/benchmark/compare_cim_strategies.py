#!/usr/bin/env python3
"""Compare CIM benchmark results across strategies and judges.

Loads checkpoint files for each (strategy, judge) pair, computes the official
CIMemories paper metrics, and prints side-by-side comparison tables.

Usage:
    uv run python src/benchmark/compare_cim_strategies.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from benchmark.metrics_cim import compute_cim_metrics  # noqa: E402

# ── CONFIGURATION ────────────────────────────────────────────────────────────
# Each strategy maps to a dict of judge_name → checkpoint path (relative to repo root).
STRATEGIES: dict[str, dict[str, str]] = {
    "Baseline": {
        "Gemini 2.5 Flash": "outputs/CIM/baseline/cim_paper_replication_gemini.json",
        "Kimi K2":          "outputs/CIM/baseline/cim_paper_replication_kimi.json",
    },
    "Defense Medium": {
        "Gemini 2.5 Flash": "outputs/CIM/defense/cim_defense_medium_gemini.json",
        "Kimi K2":          "outputs/CIM/defense/cim_defense_medium_kimi.json",
    },
    "Defense High": {
        "Gemini 2.5 Flash": "outputs/CIM/defense/cim_defense_high_gemini.json",
        "Kimi K2":          "outputs/CIM/defense/cim_defense_high_kimi.json",
    },
    "Partitioned": {
        "Gemini 2.5 Flash": "outputs/CIM/partitioned/cim_partitioned_gemini.json",
        "Kimi K2":          "outputs/CIM/partitioned/cim_partitioned_kimi.json",
    },
}

JUDGE_ORDER = ["Gemini 2.5 Flash", "Kimi K2"]
# ─────────────────────────────────────────────────────────────────────────────


def _load(path: Path) -> dict | None:
    if not path.exists():
        print(f"  [SKIP] File not found: {path}")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _fmt(val: float, std: float) -> str:
    return f"{val:5.2f} +/- {std:4.2f}"


def main() -> None:
    # Collect metrics: metrics[strategy][judge] = dict
    metrics: dict[str, dict[str, dict | None]] = {}

    for strategy, judges in STRATEGIES.items():
        metrics[strategy] = {}
        for judge, rel_path in judges.items():
            path = _PROJECT_ROOT / rel_path
            data = _load(path)
            if data is None:
                metrics[strategy][judge] = None
            else:
                metrics[strategy][judge] = compute_cim_metrics(data)

    judges = JUDGE_ORDER

    # ── Main Comparison Table ────────────────────────────────────────────
    print()
    print("=" * 110)
    print(f"{'CIM Benchmark: Strategy x Judge Comparison':^110}")
    print(f"{'Generator: Llama 3.3 70B | Metrics: Official CIMemories Paper Aggregation':^110}")
    print("=" * 110)
    print()

    # Header
    print(f"{'Strategy':<18} | {'Judge':<18} | {'Violation %':>14} | {'Coverage %':>14} | {'Entries':>8}")
    print("-" * 110)

    for strategy in STRATEGIES:
        first = True
        for judge in judges:
            m = metrics[strategy].get(judge)
            strat_col = strategy if first else ""
            first = False

            if m is None:
                print(f"{strat_col:<18} | {judge:<18} | {'--':>14} | {'--':>14} | {'--':>8}")
                continue

            viol = _fmt(m["violation_mean"], m["violation_std"])
            cov = _fmt(m["coverage_mean"], m["coverage_std"])
            print(f"{strat_col:<18} | {judge:<18} | {viol:>14} | {cov:>14} | {m['n_entries']:>8}")

        # Delta row if both judges present
        g = metrics[strategy].get(judges[0])
        k = metrics[strategy].get(judges[1])
        if g and k:
            dv = g["violation_mean"] - k["violation_mean"]
            dc = g["coverage_mean"] - k["coverage_mean"]
            print(f"{'':<18} | {'  delta (G - K)':<18} | {dv:>+10.2f}      | {dc:>+10.2f}      | {'':>8}")
        print("-" * 110)

    # ── Defense Effectiveness (per judge) ────────────────────────────────
    print()
    print("=" * 110)
    print(f"{'Defense Effectiveness vs Baseline':^110}")
    print("=" * 110)

    for judge in judges:
        print(f"\n  Judge: {judge}")
        print(f"  {'Strategy':<18} | {'Violation %':>14} | {'Coverage %':>14} | {'VR Reduction':>14} | {'Cov Change':>12}")
        print(f"  {'-' * 88}")

        baseline = metrics["Baseline"].get(judge)
        if baseline is None:
            print(f"  {'(baseline missing)':>60}")
            continue

        for strategy in STRATEGIES:
            m = metrics[strategy].get(judge)
            if m is None:
                print(f"  {strategy:<18} | {'--':>14} | {'--':>14} | {'--':>14} | {'--':>12}")
                continue

            viol = _fmt(m["violation_mean"], m["violation_std"])
            cov = _fmt(m["coverage_mean"], m["coverage_std"])

            if strategy == "Baseline":
                reduction = "--"
                cov_delta = "--"
            else:
                if baseline["violation_mean"] > 0:
                    r = (1 - m["violation_mean"] / baseline["violation_mean"]) * 100
                    reduction = f"{r:+.0f}%"
                else:
                    reduction = "N/A"
                cd = m["coverage_mean"] - baseline["coverage_mean"]
                cov_delta = f"{cd:+.1f}pp"

            print(f"  {strategy:<18} | {viol:>14} | {cov:>14} | {reduction:>14} | {cov_delta:>12}")

    # ── Per-User Violation Breakdown ─────────────────────────────────────
    print()
    print("=" * 110)
    print(f"{'Per-User Violation Rate (%) by Strategy':^110}")
    print("=" * 110)

    # Collect all user names
    all_users: set[str] = set()
    for strategy in STRATEGIES:
        for judge in judges:
            m = metrics[strategy].get(judge)
            if m and m.get("per_user_violation"):
                all_users.update(m["per_user_violation"].keys())
    users_sorted = sorted(all_users)

    for judge in judges:
        print(f"\n  Judge: {judge}")
        strat_names = list(STRATEGIES.keys())
        header = f"  {'User':<20}" + "".join(f" | {s:>16}" for s in strat_names)
        print(header)
        print(f"  {'-' * (22 + 19 * len(strat_names))}")

        for user in users_sorted:
            row = f"  {user:<20}"
            for strategy in strat_names:
                m = metrics[strategy].get(judge)
                if m and m.get("per_user_violation") and user in m["per_user_violation"]:
                    val = m["per_user_violation"][user] * 100
                    row += f" | {val:>15.2f}%"
                else:
                    row += f" | {'--':>16}"
            print(row)

    print()
    print("Violation = % of private attributes leaked  (lower is better)")
    print("Coverage  = % of required attributes included (higher is better)")
    print("Metrics use official CIMemories paper aggregation (worst-case violation, average-case coverage)")
    print()


if __name__ == "__main__":
    main()
