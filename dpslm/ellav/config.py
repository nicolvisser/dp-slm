from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from simple_parsing import Serializable


@dataclass
class TransformerModelArgs(Serializable):
    dim: int
    n_layers: int
    hidden_dim: int
    head_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    rope_theta: float


@dataclass
class ELLAVModelArgs(Serializable):
    # ------------------------------------ phonemes ------------------------------------
    num_phoneme_token_types: int
    phoneme_tokens_rate: int
    # ------------------------------------- codec --------------------------------------
    num_codec_token_types: int
    codec_tokens_rate: int
    # -----------------------=---------- transformer ----------=------------------------
    transformer: TransformerModelArgs = field(default_factory=TransformerModelArgs)


@dataclass
class ELLAVTrainArgs(Serializable):
    # -------------------------------------- wandb -------------------------------------
    project_name: str
    run_name: str
    # ----------------------------------- dataloader -----------------------------------
    phoneme_tokens_train_dir: str
    phoneme_tokens_val_dir: str
    codec_tokens_train_dir: str
    codec_tokens_val_dir: str
    batch_size: int
    num_workers: int
    # ------------------------------------ optimizer -----------------------------------
    lr_init: float
    warmup_steps: int
    lr_max: float
    decay_steps: int
    lr_final: float
    betas: Tuple[float, float]
    weight_decay: float
    eps: float
    # ----------------------------------- pl.trainer -----------------------------------
    accelerator: str
    strategy: str
    devices: int
    precision: str
    fast_dev_run: bool
    max_steps: int
    val_check_interval: float
    check_val_every_n_epoch: int
    log_every_n_steps: int
    accumulate_grad_batches: int
    gradient_clip_algorithm: Optional[str]
    gradient_clip_val: Optional[float]
    early_stopping_patience: Optional[int]
