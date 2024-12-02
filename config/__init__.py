import argparse
from pathlib import Path

import yaml
from easydict import EasyDict
from enum import Enum
from pathlib import Path
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import Any, List, Optional, Union


@dataclass
class GeneralConfig:
    train: bool = True
    test: bool = False
    gpus: int = 0
    log_wandb: bool = False
    seed: int = 0
    log_pth: Optional[Path] = None
    run_id: Optional[str] = None
    joint_segmentation: bool = False

@dataclass
class TransformConfig:
    # tf_library: TransformLibrary = TransformLibrary.torchio
    train: dict[str, Any] = field(default_factory=dict)
    test: dict[str, Any] = field(default_factory=dict)

@dataclass
class DataConfig:
    parent_path: Path = MISSING
    label_csv: Optional[str] = "table_train+test_clinical_new.csv"
    cache_dir: Optional[Path] = None
    cached_dataset: bool = False
    sequences_to_use: List[str] = field(default_factory=lambda: ["t1c", "fla", "seg"])
    target_labels: List[str] = field(default_factory=lambda: ["LF","LC"])
    use_volume: bool = False
    use_tabular: bool = True
    num_intervals: int = 20
    use_mask: bool = False
    
@dataclass
class SplitConfig:
    precomputed_splits: Optional[str] = None
    train: Optional[Union[int, float]] = None
    cross_val: Optional[int] = 5

@dataclass
class LoaderConfig:
    num_workers: int = 8
    batch_size: dict[str, Union[int,str]] = field(default_factory=lambda: {"train": 10, "test": 10})
    shuffle: dict[str, bool] = field(default_factory=lambda: {"train": True, "test": False})
    stratified_batch: bool = True
    oversampling_ones: bool = False
    same_size_batching: bool = False


class MetricNames(Enum):
    none = 0
    mcc = 1
    auroc = 2
    accuracy = 3
    f1 = 4

@dataclass
class ModelConfig:
    name: str = MISSING
    n_input_channels: int = 3
    num_target_classes: List[int] = field(default_factory=lambda: [1])
    num_tabular_features: int = 0
    additional_args: dict[str, Any] = field(default_factory=dict)

@dataclass
class LossConfig:
    loss_instances: dict[str, Any] = MISSING
    loss_weights: List[float] = MISSING
    repeat_inputs: bool = False

@dataclass
class SchedulerConfig:
    initializer: dict[str, Any] = MISSING
    monitor: str = "loss/val"
    frequency: int = 1

@dataclass
class TrainParamsConfig:
    num_epochs: int = MISSING
    grad_acc_steps: int = 1
    continue_epoch: Optional[Union[int,str]] = None
    continue_fold: int = 0
    continue_ckpt: Optional[Path] = None
    ckpt_frequency: int = 50
    overfit: Optional[int] = None
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: dict[str, Any] = MISSING
    scheduler: Optional[SchedulerConfig] = None
    early_stopping: Optional[dict[str, Any]] = None
    start_second_stage_epoch: int = 10000
    train_stages: Optional[dict[str, Any]] = None
    opt_args: dict[str, Any] = field(default_factory=dict)
    # placeholder attributes to be set later
    joint_segmentation: bool = False
    len_train_loader: Optional[int] = None
    len_val_loader: Optional[int] = None
    fold: Optional[int] = None
    
@dataclass
class TestParamsConfig:
    test_epoch: Optional[List] = field(default_factory=lambda: [None])


@dataclass
class Config:
    name: str = MISSING
    general: GeneralConfig = field(default_factory=GeneralConfig)
    datasets: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    transforms: TransformConfig = field(default_factory=TransformConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainParamsConfig = field(default_factory=TrainParamsConfig)
    # wandb: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    testing: TestParamsConfig = field(default_factory=TestParamsConfig)
    # placeholder attributes to be set later in init()
    task: str = "cox_survival"
    
    



def load_config_store():
    configstore = ConfigStore.instance()
    configstore.store(name="base_config", node=Config)


