"""Small backend protocol so experiments can supply a local model implementation."""

from pathlib import Path
from typing import Any, Protocol, Sequence

from ..config import TuningConfig
from ..data import Sample


class TrainingBackend(Protocol):
    name: str

    def predict(self, prompts: Sequence[str]) -> list[str]: ...
    def snapshot(self) -> Any: ...
    def restore(self, state: Any) -> None: ...
    def train(
        self, samples: Sequence[Sample], config: TuningConfig, *, max_steps: int, deadline: float
    ) -> dict: ...
    def save(self, directory: Path) -> None: ...
    def describe(self) -> dict: ...
