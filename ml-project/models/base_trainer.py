from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
from abc import ABC, abstractmethod


@dataclass
class TrainingResult:
    """Standard result returned by all model trainers."""
    model_dir: str
    metrics: Dict[str, Dict[str, float]]


class BaseTrainer(ABC):
    """Interface for all model trainers."""

    @abstractmethod
    def train(self) -> TrainingResult:
        """Train the model and return metrics and model location."""
        raise NotImplementedError
