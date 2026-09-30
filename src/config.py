from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: Path = Path("data/processed/source_independent")
    results_root: Path = Path("results/runs")
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 0
    epochs: int = 25
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.5
    early_stopping_patience: int = 5
    scheduler_patience: int = 2
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6
    seed: int = 42
    pretrained: bool = False
    color_jitter: bool = True

    @property
    def device(self):
        return "cuda" if __import__("torch").cuda.is_available() else "cpu"
