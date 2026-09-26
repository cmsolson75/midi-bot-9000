import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    vocab_size: int = 0  # Filled from the saved tokenizer.
    dim: int = 256
    layers: int = 10
    heads: int = 8
    kv_heads: int = 4
    hidden_dim: int = 704
    context_length: int = 1024
    dropout: float = 0.1
    rope_theta: float = 10000.0

    def validate(self):
        if (
            min(
                self.vocab_size,
                self.dim,
                self.layers,
                self.heads,
                self.kv_heads,
                self.hidden_dim,
                self.context_length,
            )
            < 1
        ):
            raise ValueError("Model dimensions and vocabulary must be positive")
        if self.dim % self.heads or self.heads % self.kv_heads:
            raise ValueError("dim must divide into heads; heads must divide into kv_heads")
        if (self.dim // self.heads) % 2:
            raise ValueError("RoPE requires an even head dimension")
        if not 0 <= self.dropout < 1 or self.rope_theta <= 0:
            raise ValueError("Invalid dropout or RoPE theta")


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    batch_size: int = 8
    accumulation_steps: int = 4
    max_steps: int = 20000
    learning_rate: float = 0.0003
    min_lr_ratio: float = 0.1
    warmup_steps: int = 500
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 250
    eval_batches: int = 32
    log_interval: int = 10
    transpose: int = 5
    augmentation_probability: float = 0.0
    velocity_shift_bins: int = 0
    duration_scale_min: float = 1.0
    duration_scale_max: float = 1.0
    eval_train_batches: int = 0
    seed: int = 42
    device: str = "auto"
    precision: str = "auto"

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        values["model"] = ModelConfig(**values.get("model", {}))
        result = cls(**values)
        for key in (
            "batch_size",
            "accumulation_steps",
            "max_steps",
            "eval_interval",
            "eval_batches",
            "log_interval",
        ):
            if getattr(result, key) < 1:
                raise ValueError(f"{key} must be positive")
        if result.warmup_steps < 0 or result.transpose < 0:
            raise ValueError("warmup_steps and transpose must be nonnegative")
        if result.learning_rate <= 0 or not 0 <= result.min_lr_ratio <= 1:
            raise ValueError("Invalid learning rate")
        if not 0 <= result.augmentation_probability <= 1:
            raise ValueError("augmentation_probability must be between 0 and 1")
        if not isinstance(result.velocity_shift_bins, int) or result.velocity_shift_bins < 0:
            raise ValueError("velocity_shift_bins must be a nonnegative integer")
        if not (
            math.isfinite(result.duration_scale_min)
            and math.isfinite(result.duration_scale_max)
            and 0 < result.duration_scale_min <= result.duration_scale_max
        ):
            raise ValueError("duration scales must be finite, positive and ordered")
        if not isinstance(result.eval_train_batches, int) or result.eval_train_batches < 0:
            raise ValueError("eval_train_batches must be a nonnegative integer")
        return result

    def to_dict(self):
        return asdict(self)
