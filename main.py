from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torch import load
from pathlib import Path
import hydra
import pandas as pd
import wandb

from omegaconf import OmegaConf
from config import Config, load_config_store #get_config
from data import get_datasets, get_dataloaders
from util import init, wandblogger, TrainingBase, CoxTrainingBase
import models

load_config_store()


def train(cfg, train_loader, val_loader, model, fold=None):
    if "survival" in cfg.task:    
        training_base = CoxTrainingBase
    else:
        training_base = TrainingBase

    trainer_opts = {
        "max_epochs": cfg.training.num_epochs,
        "log_every_n_steps": 1,
        "logger": False,
        "accelerator": "gpu" if cfg.general.gpus else "cpu",
        "check_val_every_n_epoch": cfg.training.scheduler.frequency,
        "accumulate_grad_batches": cfg.training.grad_acc_steps,
        "callbacks": [],
        "num_sanity_val_steps": 0
    }
    if cfg.general.gpus:  # 0 for cpu
        trainer_opts["devices"] = cfg.general.gpus
        if cfg.general.gpus > 1:
            trainer_opts["strategy"] = "ddp"

    cfg.training.len_train_loader = len(train_loader)
    cfg.training.len_val_loader = len(val_loader)

    log_path = cfg.general.log_pth / f"fold-{fold}" if fold is not None else cfg.general.log_pth
    log_path.mkdir(exist_ok=True)
    load_path = None
    if cfg.training.continue_epoch:
        if cfg.training.continue_ckpt:
            load_path = cfg.training.continue_ckpt / (f"fold-{fold}" if fold is not None else "") /"checkpoints"
        else:
            load_path = log_path / "checkpoints"
        cfg.training.fold = fold
        if str(cfg.training.continue_epoch).isdigit():
            load_path = load_path / f"epoch={cfg.training.continue_epoch}.ckpt"
        else:
            load_path = load_path / "best_model.ckpt"
        
        ckpt = load(load_path)
        state_dict = {k[6:]: v for k, v in ckpt["state_dict"].items()}
        model.load_state_dict(state_dict,strict=False) 


        if cfg.training.continue_ckpt:
            # if we load model from different experiment, treat only as pretraining 
            # # and don't take trainer parameters - so remove the load path
            load_path = None 
            
    base = training_base(cfg, model)

    if cfg.general.log_wandb:
        if fold and (fold > 0):
            wandb.finish()  # so that wandb does not override different fold results under same run
            # maybe more optimal way exists -- no, overrides if the run is not killed

        logger = wandblogger(cfg, log_path, fold=fold)
        trainer_opts["logger"] = logger

    if cfg.training.early_stopping is not None:
        trainer_opts["callbacks"].append(EarlyStopping(**cfg.training.early_stopping))

    trainer_opts["callbacks"].extend(
        [
            ModelCheckpoint(
                dirpath=log_path / "checkpoints",
                filename=f"best_model",
                save_top_k=1,
                verbose=True,
                monitor=cfg.training.scheduler.monitor,
                mode="max" if "loss" not in cfg.training.scheduler.monitor else "min",
            ),
            ModelCheckpoint(
                dirpath=log_path / "checkpoints",
                filename="{epoch}",
                verbose=True,
                every_n_epochs=cfg.training.ckpt_frequency,
                save_top_k=-1,
            ),
            * ([LearningRateMonitor(logging_interval="epoch")]if cfg.general.log_wandb else []),
        ]
    )

    trainer = Trainer(**trainer_opts)
    trainer.fit(base, train_loader, val_loader, ckpt_path=load_path)


def test(cfg, test_loader, model, fold=None, test_epoch=None):
    assert (cfg.general.run_id is not None), "run_id must be specified for prediction"
    
    if "survival" in cfg.task:    
        training_base = CoxTrainingBase
    else:
        training_base = TrainingBase

    log_path = cfg.general.log_pth / f"fold-{fold}" if fold is not None else cfg.general.log_pth
    if str(test_epoch).isdigit():
        ckpt_name = f"epoch={test_epoch}.ckpt"
    else:
        ckpt_name = f"best_model.ckpt"
    load_path = log_path / "checkpoints" / ckpt_name
    base = training_base.load_from_checkpoint(
        load_path,
        cfg=cfg,
        model=model,
    )
    trainer_opts = {
        "accelerator": "gpu" if cfg.general.gpus else "cpu",
        "devices": 1 if cfg.general.gpus else None,
        ##"logger": False # just write on csv (maybe improve later & do with wandb)
    }
    # will use the run that is initialized in train()
    # if we are not training, re-init
    if cfg.general.log_wandb:
        if (not cfg.general.train and fold and (fold > 0)) or (len(cfg.testing.test_epoch) > 1 and test_epoch):
            wandb.finish()  # so that wandb does not override different fold results under same run
           

        logger = wandblogger(cfg, log_path, fold=fold, test_epoch=test_epoch)
        trainer_opts["logger"] = logger

    trainer = Trainer(**trainer_opts)
    return trainer.test(base, test_loader, ckpt_path=load_path)


def validate(cfg, val_loader, model, fold=None, test_epoch=None):
    if "survival" in cfg.task:    
        training_base = CoxTrainingBase
    else:
        training_base = TrainingBase


    log_path = cfg.general.log_pth / f"fold-{fold}" if fold is not None else cfg.general.log_pth
    if str(test_epoch).isdigit():
        ckpt_name = f"epoch={test_epoch}.ckpt"
    else:
        ckpt_name = f"best_model.ckpt"
    load_path = log_path / "checkpoints" / ckpt_name
    base = training_base.load_from_checkpoint(
        load_path,
        cfg=cfg,
        model=model,
    )
    trainer_opts = {
        "accelerator": "gpu" if cfg.general.gpus else "cpu",
        "devices": 1 if cfg.general.gpus else None,
        "logger": False,
    }

    if cfg.general.log_wandb:
        # will use the run that is initialized in train()
        # to see the best val score on WandB browser
        logger = wandblogger(cfg, log_path, fold=fold, test_epoch=test_epoch)
        trainer_opts["logger"] = logger

    trainer = Trainer(**trainer_opts)
    return trainer.validate(base, val_loader, ckpt_path=load_path)

@hydra.main(version_base=None, config_path=str(Path.cwd() / "config"))
def main(cfg: Config):
    # cfg = get_config()
    # function to control the config, set up filenames, create ckpt folders and stuff
    cfg = init(cfg)
    train_val_data, test_data, train_transforms, test_transforms = get_datasets(cfg)

    if not cfg.split.cross_val:
        model_cfg = cfg.model.copy()
        model = getattr(models, model_cfg.name)(
                n_input_channels=model_cfg.n_input_channels,
                num_target_classes=model_cfg.num_target_classes,
                num_tabular_features=model_cfg.num_tabular_features,
                use_seg = "seg" in cfg.datasets.sequences_to_use,
                **model_cfg.additional_args,
            ) 

        train_loader, val_loader, test_loader = get_dataloaders(
            cfg, train_val_data, test_data, train_transforms, test_transforms
        )
        if cfg.general.train:
            train(cfg, train_loader, val_loader, model)
            val_results = None
            if not cfg.training.joint_segmentation and (cfg.training.start_second_stage_epoch is None):
                val_results = validate(cfg, val_loader, model)
                
        if cfg.general.test:
            for test_epoch in cfg.testing.test_epoch:  
                test(cfg, test_loader, model, test_epoch=test_epoch)

        if val_results:
            return val_results # return for hyperparameter optimization with sweeps
    else:
        print(f"{cfg.split.cross_val}-fold Crosss Validation")
        best_val_results = []
        best_test_results = []
        for k in range(cfg.training.continue_fold, cfg.split.cross_val):
            model_cfg = cfg.model.copy()
            model = getattr(models, model_cfg.name)(
                n_input_channels=model_cfg.n_input_channels,
                num_target_classes=model_cfg.num_target_classes,
                num_tabular_features=model_cfg.num_tabular_features,
                **model_cfg.additional_args,
            ) 

            train_loader, val_loader, test_loader = get_dataloaders(
                cfg, train_val_data, test_data, train_transforms, test_transforms, k
            )
            if cfg.general.train:
                train(cfg, train_loader, val_loader, model, k)
                if not cfg.training.joint_segmentation and (cfg.training.start_second_stage_epoch is None): 
                    val_results = validate(cfg, val_loader, model, k)
        

            if cfg.general.test:
                
                for test_epoch in cfg.testing.test_epoch:
                    test(cfg, test_loader, model, k, test_epoch=test_epoch)
                
                
        
            
if __name__ == "__main__":
    main()