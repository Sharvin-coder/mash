"""PersistBench dataset adapter."""

from pathlib import Path
from typing import Iterator

from benchmark.config import resolve_entry_configuration
from benchmark.dataset_loaders import Sample
from benchmark.utils import generate_hash_id
from benchmark.work_planner import load_input_file


class PersistBenchDataset:
    """Loads PersistBench entries from a JSON/JSONL file and yields Samples."""

    def __init__(self, input_file: Path) -> None:
        self._entries = load_input_file(input_file)

    def __iter__(self) -> Iterator[Sample]:
        for entry in self._entries:
            memories = entry["memories"]
            query = entry["query"]
            hash_id = generate_hash_id(memories, query)
            failure_type = resolve_entry_configuration(entry)

            yield Sample(
                sample_id=hash_id,
                prompt=query,
                memories=memories,
                metadata={"failure_type": failure_type},
            )
