from dataclasses import dataclass, field
from typing import List


@dataclass
class CareConfig:
    """Configuration for CARE-GNN training and model hyperparameters."""

    # dataset and model
    data: str = 'yelp'
    model: str = 'CARE'
    inter: str = 'GNN'
    batch_size: int = 1024

    # hyperparameters
    lr: float = 0.01
    lambda_1: float = 2.0
    lambda_2: float = 1e-3
    emb_size: int = 64
    num_epochs: int = 31
    test_epochs: int = 3
    under_sample: int = 1
    step_size: float = 2e-2
    dropout: float = 0.6
    leaky_relu_slope: float = 0.2

    # RL thresholds
    initial_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    threshold_min: float = 0.001
    threshold_max: float = 0.999

    # RL terminal condition
    rl_patience: int = 5
    rl_epsilon: float = 1e-4

    # training
    seed: int = 72
    device: str = 'auto'
    num_workers: int = 0

    # early stopping and checkpointing
    patience: int = 10
    checkpoint_dir: str = 'checkpoints'
    save_best: bool = True

    # logging
    log_dir: str = 'logs'
    use_tensorboard: bool = False

    # validation
    test_size: float = 0.60
    val_size: float = 0.15
    use_validation: bool = True

    # gradient clipping
    grad_clip: float = 1.0

    # learning rate scheduler
    use_lr_scheduler: bool = True
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5

    # GraphSAGE baseline
    sage_num_samples: int = 5

    # cross-validation
    num_folds: int = 5
    use_cross_validation: bool = False
