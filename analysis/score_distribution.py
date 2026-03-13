#!/usr/bin/env python3
"""Show score distribution per model per failure type.

Usage:
    python analysis/score_distribution.py outputs/my_run.json
    python analysis/score_distribution.py outputs/a.json outputs/b.json --plot dist.png
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict
from statistics import mean, median

from rich.console import Console
from rich.table import Table
from rich import box

# Scores that indicate a negative outcome for each failure type.
# cross_domain / sycophancy: higher score = more failure (1 = clean, 5 = definite failure)
# beneficial_memory_usage:   higher score = better use of memories (inverted)
INVERTED_TYPES = {"beneficial_memory_usage"}

SCORE_RANGE = list(range(1, 6))  # 1-5


def load_data(files: list[pathlib.Path]) -> dict[str, dict[str, dict[str, list[int]]]]:
    """Return failure_type -> model -> entry_id -> [valid scores]."""
    data: dict[str, dict[str, dict[str, list[int]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for path in files:
        if not path.exists():
            print(f"Warning: {path} not found, skipping", file=sys.stderr)
            continue
        with open(path) as f:
            checkpoint = json.load(f)
        for entry_id, entry in checkpoint.get("entries", {}).items():
            ft = entry.get("failure_type") or entry.get("leakage_type", "cross_domain")
            if ft == "positive_memory_usage":
                ft = "beneficial_memory_usage"
            for model, mdata in entry.get("results", {}).items():
                scores = [
                    g["judge"]["score"]
                    for g in mdata.get("generations", [])
                    if not g.get("error") and g.get("judge") and g["judge"].get("score") is not None
                ]
                prev = data[ft][model].get(entry_id, [])
                if len(scores) > len(prev):
                    data[ft][model][entry_id] = scores
    return dict(data)


def compute_distribution(
    scores_by_entry: dict[str, list[int]]
) -> tuple[dict[int, int], list[int], int]:
    """Return (score_counts, flat_score_list, total_entries_with_scores)."""
    counts: dict[int, int] = {s: 0 for s in SCORE_RANGE}
    all_scores: list[int] = []
    total = 0
    for scores in scores_by_entry.values():
        if scores:
            total += 1
        for s in scores:
            counts[s] = counts.get(s, 0) + 1
            all_scores.append(s)
    return counts, all_scores, total


def print_tables(
    data: dict[str, dict[str, dict[str, list[int]]]],
    console: Console,
) -> None:
    for ft in sorted(data.keys()):
        inverted = ft in INVERTED_TYPES
        direction = "higher=better" if inverted else "lower=better"

        console.print(f"\n[bold cyan]{'=' * 80}[/bold cyan]")
        console.print(
            f"[bold]{ft}[/bold]  |  scores 1-5  ({direction})"
        )
        console.print(f"[bold cyan]{'=' * 80}[/bold cyan]")

        table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
        table.add_column("Model", style="cyan", min_width=30)
        table.add_column("N", justify="right", min_width=5)
        for s in SCORE_RANGE:
            table.add_column(f"Score {s}", justify="right", min_width=9)
        table.add_column("Mean", justify="right", min_width=6)
        table.add_column("Median", justify="right", min_width=7)

        for model in sorted(data[ft].keys()):
            counts, all_scores, total = compute_distribution(data[ft][model])
            if not all_scores:
                continue
            n_scores = len(all_scores)
            row = [model, str(total)]
            for s in SCORE_RANGE:
                c = counts.get(s, 0)
                pct = c / n_scores * 100 if n_scores else 0
                row.append(f"{c} ({pct:.0f}%)")
            row.append(f"{mean(all_scores):.2f}")
            row.append(f"{median(all_scores):.1f}")
            table.add_row(*row)

        console.print(table)


def plot_distributions(
    data: dict[str, dict[str, dict[str, list[int]]]],
    out_path: pathlib.Path,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    failure_types = sorted(data.keys())
    n_types = len(failure_types)
    fig, axes = plt.subplots(1, n_types, figsize=(6 * n_types, 5), squeeze=False)

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for col, ft in enumerate(failure_types):
        ax = axes[0][col]
        models = sorted(data[ft].keys())
        x = np.arange(len(SCORE_RANGE))
        bar_width = 0.8 / max(len(models), 1)

        for i, model in enumerate(models):
            counts, all_scores, _ = compute_distribution(data[ft][model])
            n = len(all_scores) or 1
            heights = [counts.get(s, 0) / n * 100 for s in SCORE_RANGE]
            short_name = model.split("/")[-1][:25]
            ax.bar(
                x + i * bar_width - (len(models) - 1) * bar_width / 2,
                heights,
                width=bar_width,
                label=short_name,
                color=colors[i % len(colors)],
                alpha=0.85,
            )

        ax.set_title(ft.replace("_", " "), fontsize=11, fontweight="bold")
        ax.set_xlabel("Score")
        ax.set_ylabel("% of generations")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in SCORE_RANGE])
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Score Distribution by Failure Type and Model", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score distribution per model per failure type")
    parser.add_argument("files", nargs="+", type=pathlib.Path, help="Checkpoint JSON files")
    parser.add_argument("--plot", type=pathlib.Path, default=None, metavar="FILE",
                        help="Save bar chart to this PNG path")
    args = parser.parse_args()

    data = load_data(args.files)
    if not data:
        print("No data found", file=sys.stderr)
        sys.exit(1)

    console = Console()
    print_tables(data, console)

    if args.plot:
        plot_distributions(data, args.plot)


if __name__ == "__main__":
    main()
