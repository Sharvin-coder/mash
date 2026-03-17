#!/usr/bin/env python3
"""Inspect individual benchmark entries: query, memories, responses, scores, judge reasoning.

Useful for qualitative review — understand *why* a model scored the way it did.

Usage examples:
    # Show 5 worst-performing entries (highest failure scores) for cross_domain
    python analysis/entry_inspector.py outputs/my_run.json --failure-type cross_domain --sort worst --limit 5

    # Show entries where llama scored 5 (definite failure)
    python analysis/entry_inspector.py outputs/my_run.json --model llama --score-min 5

    # Show entries where two models disagree by 2+ score points
    python analysis/entry_inspector.py outputs/my_run.json --disagreement 2

    # Show a specific entry by its hash_id
    python analysis/entry_inspector.py outputs/my_run.json --entry-id abc123
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import defaultdict
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

INVERTED_TYPES = {"beneficial_memory_usage"}


def load_entries(files: list[pathlib.Path]) -> dict[str, dict[str, Any]]:
    """Merge entries from all files; later files may extend results."""
    merged: dict[str, dict[str, Any]] = {}
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
            if entry_id not in merged:
                merged[entry_id] = {
                    "query": entry.get("query", ""),
                    "memories": entry.get("memories", []),
                    "failure_type": ft,
                    "results": {},
                }
            for model, mdata in entry.get("results", {}).items():
                generations = mdata.get("generations", [])
                prev = merged[entry_id]["results"].get(model, {}).get("generations", [])
                if len(generations) > len(prev):
                    merged[entry_id]["results"][model] = {"generations": generations}

    return merged


def best_score(generations: list[dict]) -> int | None:
    scores = [
        g["judge"]["score"]
        for g in generations
        if not g.get("error") and g.get("judge") and g["judge"].get("score") is not None
    ]
    return max(scores) if scores else None


def mean_score(generations: list[dict]) -> float | None:
    scores = [
        g["judge"]["score"]
        for g in generations
        if not g.get("error") and g.get("judge") and g["judge"].get("score") is not None
    ]
    return sum(scores) / len(scores) if scores else None


def score_colour(score: int | None, ft: str) -> str:
    if score is None:
        return "grey50"
    inverted = ft in INVERTED_TYPES
    # For inverted (beneficial): high=good (green), low=bad (red)
    # For others: low=good (green), high=bad (red)
    if inverted:
        palette = {1: "red", 2: "dark_orange", 3: "yellow", 4: "green", 5: "bold green"}
    else:
        palette = {1: "bold green", 2: "green", 3: "yellow", 4: "dark_orange", 5: "red"}
    return palette.get(score, "white")


def format_memories(memories: list | dict) -> str:
    if isinstance(memories, dict):
        lines = []
        for cat, mems in memories.items():
            if mems:
                lines.append(f"[dim]{cat}:[/dim] " + " | ".join(mems[:3]))
        return "\n".join(lines[:8]) + ("\n…" if len(memories) > 8 else "")
    # flat list
    shown = memories[:8]
    tail = f"\n  …and {len(memories) - 8} more" if len(memories) > 8 else ""
    return "\n".join(f"  • {m}" for m in shown) + tail


def print_entry(
    entry_id: str,
    entry: dict,
    console: Console,
    model_filter: str | None,
    show_reasoning: bool,
) -> None:
    ft = entry["failure_type"]
    console.print(Panel(
        f"[bold]ID:[/bold] {entry_id}\n"
        f"[bold]Failure type:[/bold] {ft}\n"
        f"[bold]Query:[/bold] {entry['query']}",
        title="[bold cyan]Entry[/bold cyan]",
        border_style="cyan",
    ))

    # Memories
    mem_text = format_memories(entry["memories"])
    console.print(Panel(mem_text, title="[bold]Memories[/bold]", border_style="dim", expand=False))

    # Results
    for model, mdata in sorted(entry["results"].items()):
        if model_filter and model_filter.lower() not in model.lower():
            continue

        generations = mdata.get("generations", [])
        short_model = model.split("/")[-1]

        for gen in generations:
            error = gen.get("error")
            judge = gen.get("judge") or {}
            score = judge.get("score")
            reasoning = judge.get("reasoning", "")
            response = gen.get("memory_response") or ""
            gen_idx = gen.get("generation_index", 0)

            colour = score_colour(score, ft)
            score_str = f"[{colour}]{score}[/{colour}]" if score else "[grey50]—[/grey50]"

            header = f"[bold]{short_model}[/bold]  gen #{gen_idx}  score {score_str}"
            if error:
                header += f"  [red]ERROR: {error}[/red]"

            body_parts = []
            if response:
                body_parts.append(f"[dim]Response:[/dim]\n{response}")

                # if shorter response is needed
                # body_parts.append(f"[dim]Response:[/dim]\n{response[:600]}" + ("…" if len(response) > 600 else ""))

            if reasoning:
                body_parts.append(f"[dim]Judge reasoning:[/dim]\n{reasoning}")

                # if shorter response is needed
                # body_parts.append(f"[dim]Judge reasoning:[/dim]\n{reasoning[:400]}" + ("…" if len(reasoning) > 400 else ""))

            console.print(Panel(
                "\n\n".join(body_parts) if body_parts else "(no response)",
                title=header,
                border_style=colour,
                expand=True,
            ))

    console.print()


def compute_max_disagreement(entry: dict) -> int:
    """Max absolute score difference across all model pairs for this entry."""
    scores = {}
    for model, mdata in entry["results"].items():
        s = best_score(mdata.get("generations", []))
        if s is not None:
            scores[model] = s
    if len(scores) < 2:
        return 0
    vals = list(scores.values())
    return max(abs(a - b) for i, a in enumerate(vals) for b in vals[i + 1:])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect individual benchmark entries with responses and judge scores"
    )
    parser.add_argument("files", nargs="+", type=pathlib.Path, help="Checkpoint JSON files")
    parser.add_argument(
        "--failure-type", "-f",
        choices=["cross_domain", "sycophancy", "beneficial_memory_usage"],
        default=None,
        help="Filter by failure type",
    )
    parser.add_argument("--model", "-m", default=None,
                        help="Show only results for models whose name contains this string")
    parser.add_argument("--score-min", type=int, default=None,
                        help="Only show entries where best score >= this value (for any model)")
    parser.add_argument("--score-max", type=int, default=None,
                        help="Only show entries where best score <= this value (for any model)")
    parser.add_argument(
        "--sort", choices=["worst", "best", "random"], default="worst",
        help="worst = highest failure score first (default), best = lowest first"
    )
    parser.add_argument("--limit", "-n", type=int, default=10,
                        help="Max entries to show (default: 10)")
    parser.add_argument("--entry-id", default=None,
                        help="Show a specific entry by its hash_id")
    parser.add_argument("--disagreement", type=int, default=None, metavar="MIN_DELTA",
                        help="Only show entries where models disagree by at least MIN_DELTA score points")
    parser.add_argument("--show-reasoning", action="store_true",
                        help="Also print judge reasoning text")
    args = parser.parse_args()

    entries = load_entries(args.files)
    if not entries:
        print("No entries found", file=sys.stderr)
        sys.exit(1)

    # Filter
    filtered: list[tuple[str, dict]] = []
    for eid, entry in entries.items():
        if args.entry_id and eid != args.entry_id:
            continue
        if args.failure_type and entry["failure_type"] != args.failure_type:
            continue
        if args.disagreement is not None and compute_max_disagreement(entry) < args.disagreement:
            continue

        # Score filter: check any model's best score
        all_best = [
            best_score(mdata.get("generations", []))
            for mdata in entry["results"].values()
        ]
        all_best_valid = [s for s in all_best if s is not None]
        if args.model:
            # When model filter is on, only look at that model's score
            all_best_valid = [
                best_score(mdata.get("generations", []))
                for m, mdata in entry["results"].items()
                if args.model.lower() in m.lower()
                if best_score(mdata.get("generations", [])) is not None
            ]

        if not all_best_valid:
            continue
        entry_best = max(all_best_valid)
        entry_low = min(all_best_valid)

        if args.score_min is not None and entry_best < args.score_min:
            continue
        if args.score_max is not None and entry_low > args.score_max:
            continue

        filtered.append((eid, entry))

    # Sort
    def sort_key(item: tuple[str, dict]) -> float:
        _, e = item
        all_s = [
            best_score(mdata.get("generations", []))
            for mdata in e["results"].values()
        ]
        valid = [s for s in all_s if s is not None]
        return (sum(valid) / len(valid)) if valid else 0.0

    if args.sort == "worst":
        filtered.sort(key=sort_key, reverse=True)
    elif args.sort == "best":
        filtered.sort(key=sort_key, reverse=False)
    # "random" keeps insertion order

    filtered = filtered[: args.limit]

    console = Console()
    console.print(
        f"\n[bold]Showing {len(filtered)} of {len(entries)} entries[/bold]"
        + (f"  [dim](failure_type={args.failure_type})[/dim]" if args.failure_type else "")
        + (f"  [dim](sort={args.sort})[/dim]")
    )

    for eid, entry in filtered:
        print_entry(eid, entry, console, args.model, args.show_reasoning)


if __name__ == "__main__":
    main()
