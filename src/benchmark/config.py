"""Configuration loading and validation."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, PositiveInt


FAILURE_TYPE_CROSS_DOMAIN = "cross_domain"
FAILURE_TYPE_SYCOPHANCY = "sycophancy"
FAILURE_TYPE_BENEFICIAL = "beneficial_memory_usage"
FAILURE_TYPE_CIM = "cim"

VALID_FAILURE_TYPES = {
    FAILURE_TYPE_CROSS_DOMAIN,
    FAILURE_TYPE_SYCOPHANCY,
    FAILURE_TYPE_BENEFICIAL,
    FAILURE_TYPE_CIM,
}

# Per-category generation defaults when no global override is set
DEFAULT_GENERATIONS_BY_FAILURE_TYPE = {
    FAILURE_TYPE_CROSS_DOMAIN: 3,
    FAILURE_TYPE_SYCOPHANCY: 3,
    FAILURE_TYPE_BENEFICIAL: 1,
    FAILURE_TYPE_CIM: 1,
}


def get_generations_for_failure_type(
    failure_type: str,
    generations_override: int | None = None,
) -> int:
    """Resolve generation count for a failure type.

    If generations_override is set, uses that for all types.
    Otherwise uses per-category defaults (3 for cross_domain/sycophancy, 1 for cim).
    """
    if generations_override is not None:
        return generations_override
    return DEFAULT_GENERATIONS_BY_FAILURE_TYPE.get(failure_type, 3)

# Backwards compatibility aliases
_LEGACY_ALIASES = {
    "leakage_type": "failure_type",
}

JUDGE_MODEL = "moonshotai/kimi-k2-thinking-maas"  # Vertex AI model name
JUDGE_MODEL_OPENROUTER = "qwen/qwen3-235b-a22b"  # OpenRouter model name
JUDGE_MODEL_GEMINI = "gemini-2.5-flash"  # Google AI Studio model name
JUDGE_LOCATION = "global"
JUDGE_TEMPERATURE = 0.0

# Global seed for reproducibility across all providers
BENCHMARK_SEED = 42

VALID_JUDGE_PROVIDERS = {"vertexai", "openrouter", "gemini"}

# ── Evaluation strategy ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STRATEGIES_DIR = PROJECT_ROOT / "prompts" / "defensive" / "strategies"
JUDGES_DIR = PROJECT_ROOT / "prompts" / "judges"


@dataclass
class EvalStrategy:
    """Loaded evaluation strategy — routes input, system prompt, and judges."""

    name: str
    description: str
    input_file: Path
    system_prompt_path: Path
    system_prompt_content: str
    judge_prompts: dict[str, str] = field(default_factory=dict)


def list_eval_strategies(strategies_dir: Path = STRATEGIES_DIR) -> list[dict[str, str]]:
    """List available evaluation strategies from prompts/defensive/strategies/."""
    results = []
    if not strategies_dir.exists():
        return results
    for path in sorted(strategies_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            results.append({
                "name": data.get("name", path.stem),
                "description": data.get("description", ""),
            })
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def load_eval_strategy(
    name: str,
    strategies_dir: Path = STRATEGIES_DIR,
    judges_dir: Path = JUDGES_DIR,
) -> EvalStrategy:
    """Load an evaluation strategy by name.

    Reads the strategy JSON and resolves all file references:
    - ``input`` → JSONL path (relative to project root)
    - ``system_prompt`` → prompt template path (relative to project root)
    - ``judges.<failure_type>`` → judge prompt filename (resolved from judges_dir)

    For failure types not listed in the manifest, the caller falls back to the
    hardcoded defaults in prompts.py.
    """
    strategy_path = strategies_dir / f"{name}.json"
    if not strategy_path.exists():
        available = [s["name"] for s in list_eval_strategies(strategies_dir)]
        raise ValueError(
            f"Evaluation strategy '{name}' not found at {strategy_path}. "
            f"Available strategies: {available}"
        )

    data = json.loads(strategy_path.read_text(encoding="utf-8"))

    # Resolve input file
    input_file = PROJECT_ROOT / data["input"]
    if not input_file.exists():
        raise ValueError(
            f"Strategy '{name}': input file '{data['input']}' not found at {input_file}"
        )

    # Resolve system prompt
    system_prompt_path = PROJECT_ROOT / data["system_prompt"]
    if not system_prompt_path.exists():
        raise ValueError(
            f"Strategy '{name}': system_prompt '{data['system_prompt']}' "
            f"not found at {system_prompt_path}"
        )
    system_prompt_content = system_prompt_path.read_text(encoding="utf-8")

    if "{memories}" not in system_prompt_content:
        raise ValueError(
            f"Strategy '{name}': system prompt must contain the {{memories}} placeholder."
        )

    # Resolve judge prompts (filenames relative to judges_dir)
    judge_prompts: dict[str, str] = {}
    for failure_type, judge_filename in data.get("judges", {}).items():
        judge_path = judges_dir / judge_filename
        if not judge_path.exists():
            raise ValueError(
                f"Strategy '{name}': judge file '{judge_filename}' for "
                f"'{failure_type}' not found at {judge_path}"
            )
        judge_prompts[failure_type] = judge_path.read_text(encoding="utf-8")

    return EvalStrategy(
        name=data.get("name", name),
        description=data.get("description", ""),
        input_file=input_file,
        system_prompt_path=system_prompt_path,
        system_prompt_content=system_prompt_content,
        judge_prompts=judge_prompts,
    )


class ModelEntry(BaseModel):
    """Individual model entry with API parameters."""

    name: str
    provider: str = "openrouter"
    mode: str | None = "sequential"
    api_params: dict[str, Any] | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    input: Path | None = None  # Per-model input file (only used when config method="partitioned")


JUDGE_MODEL_ENTRY_OPENROUTER = ModelEntry(
    name=JUDGE_MODEL_OPENROUTER,
    api_params={
        "temperature": JUDGE_TEMPERATURE,
        "seed": BENCHMARK_SEED,
        "reasoning": {"enabled": True, "effort": "high"},
    },
)


def resolve_entry_configuration(entry: dict[str, Any]) -> str:
    """Resolve failure type with defaults. Accepts legacy 'leakage_type' field."""
    failure_type = entry.get("failure_type") or entry.get(
        "leakage_type", FAILURE_TYPE_CROSS_DOMAIN
    )
    return _LEGACY_ALIASES.get(failure_type, failure_type)


def validate_failure_type(failure_type: str) -> None:
    """Validate that failure_type is supported."""
    if failure_type not in VALID_FAILURE_TYPES:
        raise ValueError(
            f"Invalid failure_type={failure_type}. Valid values: {sorted(VALID_FAILURE_TYPES)}"
        )


class BenchmarkConfig(BaseModel):
    """Benchmark configuration schema."""

    models: list[ModelEntry] = Field(min_length=1)
    judge: ModelEntry | None = None
    judge_provider: str | None = None
    input: Path
    output: Path
    method: str | None = None  # "partitioned" enables per-model input files
    store_raw_api_responses: bool = False
    generations: PositiveInt | None = None
    concurrency: PositiveInt = 1
    limit: PositiveInt | None = None
    batch_poll_timeout_minutes: PositiveInt = 25
    prompt_template: Path | None = None

    # Dataset configuration
    dataset: str = "persistbench"
    memory_mode: str = "full_profile"
    cim_path: str | None = None
    cim_labels_file: str | None = None
    cim_judge_variant: str = "reveal_paper_compat"

    # Model overrides
    generator_model: str | None = None
    judge_model_name: str | None = None
    provider: str = "openrouter"

    # Evaluation strategy (name of a .json file in prompts/defensive/strategies/)
    eval_strategy: str | None = None

    # Loaded template content (not part of JSON schema)
    prompt_template_content: str | None = None


def load_benchmark_config_data(
    data: dict[str, Any], config_path: str | Path | None = None
) -> BenchmarkConfig:
    """Load and validate config from a parsed dict."""
    if "no_memory_baseline" in data:
        raise ValueError(
            f"Config field 'no_memory_baseline' is no longer supported ({config_path}). "
            "Single-response evaluation always generates with the provided memories."
        )

    if data.get("judge") is not None:
        warning_msg = f"WARNING: The 'judge' field in configs ({config_path}) is deprecated and will be ignored."
        print(f"\n{warning_msg}\n")
        warnings.warn(
            warning_msg,
            DeprecationWarning,
            stacklevel=2,
        )
        del data["judge"]

    config = BenchmarkConfig(**data)

    # Validate method field
    if config.method is not None and config.method != "partitioned":
        raise ValueError(
            f"Invalid method '{config.method}' in config ({config_path}). "
            f"Valid values: 'partitioned' (or omit for default behaviour)."
        )

    # Validate unique model names (results are keyed by name in checkpoint)
    model_names = [m.name for m in config.models]
    seen: set[str] = set()
    for name in model_names:
        if name in seen:
            raise ValueError(
                f"Duplicate model name '{name}' in config. "
                f"Each model must have a unique name since results are keyed by model name."
            )
        seen.add(name)

    # Validate judge_provider if specified
    if (
        config.judge_provider is not None
        and config.judge_provider not in VALID_JUDGE_PROVIDERS
    ):
        raise ValueError(
            f"Invalid judge_provider '{config.judge_provider}' in config ({config_path}). "
            f"Valid values: {sorted(VALID_JUDGE_PROVIDERS)}"
        )

    # Load prompt template content if specified and not already provided (e.g. from checkpoint)
    if config.prompt_template and not config.prompt_template_content:
        template_path = Path(config.prompt_template)
        if not template_path.exists():
            raise ValueError(
                f"Prompt template file {config.prompt_template} does not exist"
            )
        else:
            print(f"prompt template for {model_names} is {config.prompt_template}")
        with open(template_path, "r", encoding="utf-8") as f:
            config.prompt_template_content = f.read()

    if (
        config.prompt_template_content
        and "{memories}" not in config.prompt_template_content
    ):
        raise ValueError(
            "Prompt template must contain the {memories} placeholder. "
            "Without it, the model receives no user memories and evaluation is meaningless."
        )

    return config


def load_benchmark_config(config_path: str | Path) -> BenchmarkConfig:
    """Load and validate config from JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise ValueError(f"Config file {config_path} does not exist")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(
            f"Config file {config_path} must contain a JSON object at top-level"
        )
    return load_benchmark_config_data(data, config_path=path)
