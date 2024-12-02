from pathlib import Path
from numpy.random import seed as npseed
from random import seed as rseed
from torch import manual_seed
# import torch.backends.cudnn.deterministic
# from random import seed
# import numpy as np

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
# from easydict import EasyDict
from bidict import bidict
from itertools import chain

import warnings
import yaml
import monai.utils
import wandb.util
import pytorch_lightning.loggers as pl_loggers
import matplotlib.pyplot as plt

def control_config(cfg):
    

    if not set(cfg.datasets.sequences_to_use).issubset(
        {"t2", "fla", "t1", "t1c", "seg"}
    ):
        raise ValueError(
            "Sequence type not recognized. Select from {'t2', 'fla', 't1', 't1c', 'seg'}. Case sensitive!"
        )
    if hasattr(cfg.split, "train") and (
        cfg.split.train and cfg.split.cross_val
    ):  # or (hasattr(cfg.split, 'cross_val') and (cfg.split.train and cfg.split.cross_val)):
        raise ValueError(
            "K-Fold Cross Validation and a single training run are not possible at the same time. Please select one!"
        )
    
    cfg.datasets.use_mask = False
    # cfg.use_t1c_separate = False
    if "CropToMask" in list(cfg.transforms.train.keys()):
        if "CropToMask" not in list(cfg.transforms.test.keys()):
            raise ValueError(
                "If you are using CropToMask, please do in both train and test."
            )
        else:
            cfg.datasets.use_mask = True
            assert (
                cfg.transforms.train["CropToMask"]["keys"]
                == cfg.transforms.test["CropToMask"]["keys"]
            )
            # if "t1c" in cfg.transforms.train["CropToMask"].keys:
            #     cfg.use_t1c_separate = True

    if "CropToMask" in list(cfg.transforms.test.keys()):
        if "CropToMask" not in list(cfg.transforms.train.keys()):
            raise ValueError(
                "If you are using CropToMask, please do in both train and test."
            )
        else:
            cfg.datasets.use_mask = True

    loss_names = list(cfg.training.loss.loss_instances.keys())
    model_name = cfg.model.name
    
    if "CoxPHLoss" in loss_names:
        cfg.task = "cox_survival"
        if cfg.datasets.target_labels not in (["LF", "LC"], ["LRF", "LRC"], ["LeptoF", "LeptoC"], ["BF", "BC"]):
            raise ValueError(
                "Both event and time-to-event targets are required for cox regression. In [<event>,<time>] order!"
            )
        cfg.training.scheduler.monitor = "cindex/val"
    elif "DiscreteSurvLoss" in loss_names or "GensheimerLoss" in loss_names:
        cfg.task = "discrete_survival"
        cfg.training.scheduler.monitor = "cindex/val"
    else:  # normal classification/regression - will be barely used
        if not cfg.task == "autoencoder":
            label_to_num_class = bidict({"LF": 1, "LC": -1})
            model_num_classes = cfg.model.num_target_classes
            for i, label in enumerate(cfg.datasets.target_labels):
                assert (
                    model_num_classes[i] == label_to_num_class[label]
                ), "Wrong label-num_class mapping!"
            cfg.task = "classification"
            cfg.training.scheduler.monitor = "mcc/val"
        else:
            cfg.task = "autoencoder"
            # cfg.training.scheduler.monitor = "mse/val"
    
    
    if cfg.training.train_stages is not None:
        assert set(cfg.training.train_stages.keys()).issubset({"only_seg","joint","only_surv"}), "Unknown training stage"

    
    if cfg.datasets.use_tabular:
        assert cfg.model.num_tabular_features > 0, "Tabular features are used but not given in the model config"

    if "seg" in cfg.datasets.sequences_to_use:
        for name, opt in chain(
            cfg.transforms.train.items(), cfg.transforms.train.items()
        ):
            if "mask" not in opt["keys"]:
                warnings.warn(
                    f"Segmentation mask is among channels but transform '{name}' is not applied to it. Are you sure?"
                )

    if isinstance(cfg.testing.test_epoch, int) or cfg.testing.test_epoch is None:
        cfg.testing.test_epoch = [cfg.testing.test_epoch]


def save_config(cfg):
    """
    Turn nested EasyDicts & PosixPaths into yaml readable format and save
    """
    # cfg_dict = _recursive_edict2dict(cfg)

    # with open(cfg.log_pth / "config.yaml", "w") as f:
    #     yaml.dump(dict(cfg_dict), f)

    cfg_dict = _recursive_confdict2dict(cfg)
    OmegaConf.save(DictConfig(cfg_dict), cfg.general.log_pth / "config.yaml")


def _recursive_confdict2dict(cfg_dict):
    cfg_dict_new = dict(cfg_dict.copy())
    for k, v in cfg_dict.items():
        if isinstance(v, DictConfig):
            cfg_dict_new[k] = _recursive_confdict2dict(v)
        elif isinstance(v, Path):
            cfg_dict_new[k] = str(v)
        
    return cfg_dict_new


def init(cfg):
    print("#" * 40 + " Experiment: {} ".format(cfg.name) + "#" * 40)
    # seed_everything(cfg.general.seed)
    manual_seed(cfg.general.seed)
    npseed(cfg.general.seed)
    rseed(cfg.general.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    monai.utils.misc.set_determinism(seed=cfg.general.seed)

    control_config(cfg)

    # cfg.log_dir = Path(cfg.log_dir).expanduser().resolve()

    if not hasattr(cfg.general, "run_id") or not cfg.general.run_id:
        cfg.general.run_id = wandb.util.generate_id()

    cfg.general.log_pth = cfg.general.log_pth / cfg.name / cfg.general.run_id
    cfg.general.log_pth.mkdir(parents=True, exist_ok=True)
    # cfg.ckpt_dir = cfg.log_pth / "checkpoints"

    # cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("Log directory created: {}".format(cfg.general.log_pth))

    save_config(cfg)

    return cfg


def wandblogger(
    cfg: DictConfig, log_path: Path, fold: int = None, test_epoch: int = None
) -> pl_loggers.WandbLogger:
    name = None
    if fold is not None and test_epoch:
        name = f"fold-{fold}" + f"_test-{test_epoch}"
    elif fold is not None:
        name = f"fold-{fold}"
    return pl_loggers.WandbLogger(
        project="aurora_" + cfg.model.name.lower(),
        save_dir=log_path,
        name=name,
        # version="",
        log_model=False,
        group=cfg.name,  # + f"-fold-{fold}" if fold is not None else cfg.name
        config=dict(cfg),
        id=cfg.general.run_id if fold is None else None,
    )


def visualize_images(imgA,reconstructed_A,fakeB=None):
    fig,axs = plt.subplots(1,2,figsize=(15,5))
    axs[0].imshow(imgA,cmap='gray')
    axs[0].set_title('Real A')
    axs[1].imshow(reconstructed_A,cmap='gray')
    axs[1].set_title('Reconstructed A')
    # axs[2].imshow(fakeB,cmap='gray')
    # axs[2].set_title('Fake B')
    
    for ax in axs:
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_xticks([])

    return fig

def find_discordants(pred, events, durations):
    """
    Find discordant pairs in the predicted risk scores
    """
    n = len(pred)
    total_discords = []
    for i in range(n):
        discords = 0
        for j in range(n):
            if i == j:
                continue
            if events[i] == 1 and events[j] == 1:
                if durations[i] < durations[j] and pred[i] > pred[j]:
                    discords += 1
                elif durations[i] > durations[j] and pred[i] < pred[j]:
                    discords += 1
            elif events[i] == 0 and events[j] == 1:
                if durations[i] > durations[j] and pred[i] > pred[j]:
                    discords += 1
            elif events[i] == 1 and events[j] == 0:
                if durations[i] < durations[j] and pred[i] < pred[j]:
                    discords += 1
            else:
                pass
        total_discords.append(discords)
    return total_discords 