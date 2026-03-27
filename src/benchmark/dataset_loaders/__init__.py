"""Dataset abstraction for benchmark evaluation."""

from dataclasses import dataclass, field
from typing import Iterator, Protocol


@dataclass
class Sample:
    """A single evaluation sample."""

    sample_id: str
    prompt: str
    memories: list[str]
    required_attributes: list[str] = field(default_factory=list)
    forbidden_attributes: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class BaseDataset(Protocol):
    """Protocol for dataset implementations."""

    def __iter__(self) -> Iterator[Sample]: ...
