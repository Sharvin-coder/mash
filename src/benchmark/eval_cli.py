#!/usr/bin/env python3
"""Command-line interface for benchmark tool."""

import argparse
import asyncio
import os
import sys
import textwrap
from pathlib import Path

os.environ.setdefault(
    "VERTEXAI_SERVICE_ACCOUNT_PATH",
    str(Path.home() / "Downloads" / "VERTEXAI_SERVICE_ACCOUNT.json"),
)

from benchmark.benchmark_runner import (
    cancel_batch_jobs,
    run_benchmark_with_retry,
)
from benchmark.metrics_cim import run_cim_metrics_cli


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared arguments to a subcommand parser."""
    parser.add_argument(
        "file",
        help="Config file (JSON) or checkpoint file to resume from. Auto-detected.",
    )
    parser.add_argument(
        "--dry-run",
        "-d",
        action="store_true",
        help="Preview configuration without making API calls",
    )
    parser.add_argument(
        "--ignore-config-mismatch",
        action="store_true",
        help="Bypass errors when config doesn't match previous checkpointed config",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        help="Limit number of entries to process (overrides config file limit)",
    )
    parser.add_argument(
        "--batch-poll-timeout",
        type=int,
        help="Timeout in minutes for batch job polling (overrides config, default: 25)",
    )
    parser.add_argument(
        "--cancel",
        action="store_true",
        help="Cancel all active batch jobs and clear them from the checkpoint",
    )
    parser.add_argument(
        "--auto-rerun",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically rerun failed benchmarks with reduced concurrency (up to 3 retries). Use --no-auto-rerun to disable.",
    )
    parser.add_argument(
        "--judge-provider",
        choices=["vertexai", "openrouter", "gemini"],
        help="Judge provider (overrides config and env var). Default: openrouter",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        help="Number of concurrent requests (overrides config file setting)",
    )
    parser.add_argument(
        "--store-raw-api-responses",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Store full raw provider API responses in output/checkpoint files. "
        "Defaults to off to reduce output size.",
    )
    parser.add_argument(
        "--dataset",
        choices=["persistbench", "cim", "both"],
        default=None,
        help="Dataset to evaluate: persistbench, cim, or both (default: persistbench)",
    )
    parser.add_argument(
        "--memory-mode",
        choices=["none", "relevant_only", "mixed", "full_profile"],
        default=None,
        help="Memory mode for CIM dataset (default: full_profile)",
    )
    parser.add_argument(
        "--cim-path",
        default=None,
        help="HuggingFace dataset ID or local path for CIM dataset",
    )
    parser.add_argument(
        "--cim-labels",
        default=None,
        help="Path to pre-computed CIM labels file (from `benchmark cim-label`)",
    )
    parser.add_argument(
        "--cim-judge-variant",
        choices=["default", "reveal_paper_compat", "reveal_official"],
        default=None,
        help="CIM judge variant: 'default' (legacy), 'reveal_paper_compat' (REVEAL metric), or 'reveal_official' (official CIMemories REVEAL). Default: reveal_paper_compat",
    )
    parser.add_argument(
        "--auto-label",
        action="store_true",
        default=False,
        help="Automatically generate CIM labels (via cim-label) before running the benchmark "
        "if no --cim-labels file is provided or the file doesn't exist. "
        "Enables a single-command CIM benchmark run.",
    )
    parser.add_argument(
        "--generator-model",
        default=None,
        help="Override generator model name",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Override judge model name",
    )
    parser.add_argument(
        "--provider",
        choices=["openrouter", "gemini"],
        default=None,
        help="Provider for generator/judge models (default: openrouter)",
    )


def _exit_code_for_subcommand(stats, *, subcommand: str) -> int:
    """Compute CLI exit code based on mode-specific completion criteria."""
    if subcommand == "generate":
        # Generation-only: only generation failures matter (judge errors are irrelevant).
        return 1 if (stats.failed_generation > 0 or stats.pending_generation > 0) else 0
    # run/judge: require full completion (no failures and nothing pending)
    return 1 if (stats.failed > 0 or stats.pending > 0) else 0


async def _handle(args: argparse.Namespace) -> int:
    if args.cancel:
        await cancel_batch_jobs(file_path=args.file)
        return 0

    subcommand = args.subcommand
    skip_generation = subcommand == "judge"
    skip_judge = subcommand == "generate"

    # Auto-label: generate CIM labels if needed before running the benchmark
    cim_labels = args.cim_labels
    effective_dataset = args.dataset
    if effective_dataset is None:
        # Check config file for dataset setting
        from pathlib import Path as _Path
        import orjson
        _cfg_path = _Path(args.file)
        if _cfg_path.exists():
            try:
                _cfg_data = orjson.loads(_cfg_path.read_bytes())
                if isinstance(_cfg_data, dict) and "entries" not in _cfg_data:
                    effective_dataset = _cfg_data.get("dataset", "persistbench")
            except Exception:
                pass

    is_cim = effective_dataset in ("cim", "both")
    if is_cim and args.auto_label and not cim_labels:
        from pathlib import Path as _Path
        from benchmark.dataset_loaders.cim_labeler import LabelingConfig, run_labeling

        default_labels_path = _Path("outputs/cim_labels.json")
        if not default_labels_path.exists():
            print("\n--- Auto-labeling CIM attributes ---")
            label_config = LabelingConfig(
                dataset_id=args.cim_path or "facebook/CIMemories",
            )
            labels_path = await run_labeling(label_config)
            cim_labels = str(labels_path)
            print(f"--- Auto-labeling complete: {labels_path} ---\n")
        else:
            print(f"Using existing labels file: {default_labels_path}")
            cim_labels = str(default_labels_path)

    stats = await run_benchmark_with_retry(
        file_path=args.file,
        dry_run=args.dry_run,
        ignore_config_mismatch=args.ignore_config_mismatch,
        limit=args.limit,
        skip_judge=skip_judge,
        skip_generation=skip_generation,
        batch_poll_timeout_minutes=args.batch_poll_timeout,
        retry_enabled=args.auto_rerun,
        judge_provider=args.judge_provider,
        concurrency_override=args.concurrency,
        store_raw_api_responses=args.store_raw_api_responses,
        dataset=args.dataset,
        memory_mode=args.memory_mode,
        cim_path=args.cim_path,
        cim_labels=cim_labels,
        cim_judge_variant=args.cim_judge_variant,
        generator_model=args.generator_model,
        judge_model=args.judge_model,
        provider=args.provider,
    )
    return _exit_code_for_subcommand(stats, subcommand=subcommand)


async def main_async() -> int:
    """Main CLI interface for benchmark tool.

    Returns:
        Exit code (0 for success, non-zero for failures).
    """
    parser = argparse.ArgumentParser(
        description="Batch LLM Generation & Judge Evaluation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Example usage:
              benchmark run config.json                  # Full run (generation + judgment)
              benchmark run output.json                  # Resume from checkpoint
              benchmark generate config.json --dry-run   # Preview generation only
              benchmark judge output.json                # Judge existing generations
        """),
    )

    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run full benchmark (generation + judgment)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(run_parser)

    generate_parser = subparsers.add_parser(
        "generate",
        help="Run generation only (no judgment)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_arguments(generate_parser)

    judge_parser = subparsers.add_parser(
        "judge",
        help="Run judgment only on existing generations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Runs judge evaluation on all generations that have responses but are
            missing judge scores. Errors if any generations are missing responses.

            Example:
              benchmark judge output.json
              benchmark judge output.json --judge-provider openrouter --concurrency 20
        """),
    )
    _add_arguments(judge_parser)

    cim_label_parser = subparsers.add_parser(
        "cim-label",
        help="Generate persona-based labels for CIMemories dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Generates share/private labels for CIMemories attributes using
            Westin privacy personas (fundamentalist, pragmatic, unconcerned).

            Example:
              benchmark cim-label
              benchmark cim-label --model DeepSeek-V3.2-3 --provider azure --concurrency 20
              benchmark cim-label --limit 25
              benchmark cim-label --aggregate-only
        """),
    )
    cim_label_parser.add_argument(
        "--model", default="DeepSeek-V3.2-3",
        help="Model for labeling (default: DeepSeek-V3.2-3)",
    )
    cim_label_parser.add_argument(
        "--provider", choices=["openrouter", "gemini", "vertexai_oss", "vertexai", "azure"], default="azure",
        help="Provider for labeling model (default: azure)",
    )
    cim_label_parser.add_argument(
        "--concurrency", type=int, default=10,
        help="Concurrent API calls (default: 10)",
    )
    cim_label_parser.add_argument(
        "--samples", type=int, default=10,
        help="Samples per persona (default: 10)",
    )
    cim_label_parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit labeling to the first N CIM groups/samples",
    )
    cim_label_parser.add_argument(
        "--output", default="outputs/cim_labels.json",
        help="Output labels file path",
    )
    cim_label_parser.add_argument(
        "--checkpoint", default="outputs/cim_labeling_checkpoint.json",
        help="Checkpoint file for resuming labeling",
    )
    cim_label_parser.add_argument(
        "--dataset-id", default="facebook/CIMemories",
        help="HuggingFace dataset ID",
    )
    cim_label_parser.add_argument(
        "--aggregate-only", action="store_true",
        help="Skip LLM calls, just aggregate existing checkpoint into labels",
    )
    cim_label_parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Temperature for labeling calls (default: 0.7)",
    )

    cim_metrics_parser = subparsers.add_parser(
        "cim-metrics",
        help="Compute aggregate CIM metrics (violation + coverage) from checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Computes official CIMemories violation and coverage metrics from a
            checkpoint/output JSON file.

            Example:
              benchmark cim-metrics output.json
              benchmark cim-metrics output.json --model gpt-4o
        """),
    )
    cim_metrics_parser.add_argument(
        "file",
        help="Checkpoint/output JSON file to compute metrics from",
    )
    cim_metrics_parser.add_argument(
        "--model",
        default=None,
        help="Filter results to a specific model name",
    )

    args = parser.parse_args()

    if args.subcommand == "cim-metrics":
        run_cim_metrics_cli(args.file, model_name=args.model)
        return 0

    if args.subcommand == "cim-label":
        from pathlib import Path
        from benchmark.dataset_loaders.cim_labeler import LabelingConfig, run_labeling, load_cim_groups, aggregate_labels, save_labels, _load_checkpoint

        config = LabelingConfig(
            dataset_id=args.dataset_id,
            model_name=args.model,
            provider=args.provider,
            samples_per_persona=args.samples,
            concurrency=args.concurrency,
            temperature=args.temperature,
            limit=args.limit,
            checkpoint_path=Path(args.checkpoint),
            output_path=Path(args.output),
        )

        if args.aggregate_only:
            print("Aggregate-only mode: loading checkpoint and groups...")
            groups = load_cim_groups(config.dataset_id, config.split)
            checkpoint = _load_checkpoint(config.checkpoint_path)
            labels = aggregate_labels(checkpoint, groups)
            save_labels(labels, config.output_path, config)
        else:
            await run_labeling(config)
        return 0

    return await _handle(args)


def main():
    """Entry point that runs async main."""
    sys.exit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
