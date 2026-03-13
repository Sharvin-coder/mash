# Evaluation Strategies

An evaluation strategy is a single JSON file that controls the **entire benchmark pipeline** by pointing to the right files for three things:

1. **Input data** — which JSONL file to use (full benchmark or pre-partitioned memories)
2. **System prompt** — which prompt the model-under-test receives (defensive variant, permissive, etc.)
3. **Judge prompts** — which scoring rubric the judge LLM uses for each failure type

## Quick Start

```bash
# See what strategies are available
uv run benchmark strategies

# Run a full benchmark with a strategy
uv run benchmark run config.json --eval-strategy GEPA_partitioned

# Judge existing generations with a different strategy
uv run benchmark judge output.json --eval-strategy permissive_full

# You can also set it in config.json instead of using the CLI flag:
# { "eval_strategy": "GEPA_partitioned", ... }
```

## Available Strategies

| Name | Input | System Prompt | Beneficial Judge |
|------|-------|--------------|-----------------|
| `default` | `full_benchmark.jsonl` | `evaluation.txt` | Cross-domain (1-5) |
| `GEPA_partitioned` | `full_benchmark_partitioned_llama3p3_70b.jsonl` | `GEPA_optimized.txt` | Dedicated (1-3) |
| `permissive_full` | `full_benchmark.jsonl` | `permissive.txt` | Dedicated (1-3) |

## How a Strategy File Works

A strategy is a `.json` file in this directory. Here's an annotated example:

```json
{
  "name": "GEPA_partitioned",
  "description": "GEPA-optimized defensive prompt with Llama3.3-70B memory partitioning",

  "input": "benchmark_samples/full_benchmark_partitioned_llama3p3_70b.jsonl",
  //       ↑ Path to the JSONL input file (relative to project root)
  //         Use a *_partitioned_* file for pre-filtered memories,
  //         or full_benchmark.jsonl for all memories

  "system_prompt": "prompts/defensive/GEPA_optimized.txt",
  //               ↑ The prompt template sent to the model being tested
  //                 Must contain {memories} placeholder
  //                 Choose from: prompts/evaluation.txt
  //                              prompts/generic_prompt.txt
  //                              prompts/defensive/GEPA_optimized.txt
  //                              prompts/defensive/permissive.txt
  //                              prompts/defensive/restrictive.txt
  //                              prompts/defensive/rubric_informed.txt

  "judges": {
    "cross_domain": "cross_domain.txt",
    "sycophancy": "sycophancy.txt",
    "beneficial_memory_usage": "beneficial_samples.txt"
    //                         ↑ Filenames from prompts/judges/
    //                           cross_domain.txt       → 1-5 scale leakage rubric
    //                           sycophancy.txt         → 1-5 scale sycophancy rubric
    //                           beneficial_samples.txt → 1-3 scale memory usage rubric
    //
    //  You can omit a failure type to fall back to the hardcoded default.
    //  Example: omitting "sycophancy" uses the built-in sycophancy judge.
  }
}
```

## Creating a New Strategy

1. Copy an existing `.json` file in this directory
2. Rename it (the filename becomes the strategy name used in `--eval-strategy`)
3. Edit the three fields: `input`, `system_prompt`, `judges`
4. Run `uv run benchmark strategies` to verify it shows up

### Example: Restrictive prompt with Qwen3 partitioning

Create `restrictive_qwen3.json`:

```json
{
  "name": "restrictive_qwen3",
  "description": "Restrictive defensive prompt with Qwen3-235B memory partitioning",
  "input": "benchmark_samples/beneficial_samples_partitioned_qwen3_235b.jsonl",
  "system_prompt": "prompts/defensive/restrictive.txt",
  "judges": {
    "cross_domain": "cross_domain.txt",
    "sycophancy": "sycophancy.txt",
    "beneficial_memory_usage": "beneficial_samples.txt"
  }
}
```

Then run:
```bash
uv run benchmark run config.json --eval-strategy restrictive_qwen3
```

## Adding a Custom Judge Prompt

If you want a strategy to use a completely new scoring rubric:

1. Write your judge prompt as a `.txt` file in `prompts/judges/`
2. The prompt must instruct the judge LLM to return JSON with `"reasoning"` and `"score"` (or `"rating"`) keys
3. Reference the new filename in your strategy's `judges` section

## File Locations

```
prompts/
├── defensive/
│   ├── GEPA_optimized.txt          ← System prompts for model-under-test
│   ├── permissive.txt
│   ├── restrictive.txt
│   ├── rubric_informed.txt
│   └── strategies/                 ← Strategy files (YOU ARE HERE)
│       ├── README.md
│       ├── default.json
│       ├── GEPA_partitioned.json
│       └── permissive_full.json
├── judges/                         ← Judge scoring rubrics (shared)
│   ├── cross_domain.txt              1-5 scale
│   ├── sycophancy.txt                1-5 scale
│   └── beneficial_samples.txt        1-3 scale
├── evaluation.txt                  ← Default system prompt (ChatGPT-style)
└── generic_prompt.txt              ← Minimal system prompt
```
