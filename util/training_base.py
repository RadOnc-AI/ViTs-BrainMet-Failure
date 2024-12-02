from itertools import islice
from typing import Union
from random import randint

from torch.optim.optimizer import Optimizer
from easydict import EasyDict
from pandas import DataFrame

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import grad_norm
from torch import optim, nn, tensor, cat, mean, sigmoid, no_grad, Tensor, load
from monai.data.meta_tensor import MetaTensor
from monai.data import decollate_batch
from wandb import plot
from copy import deepcopy

from loss import CombinedLoss
from data import transforms
from util.util import visualize_images
import metric

# import pandas as pd


class TrainingBase(LightningModule):
    """
    Pytorch Lightning wrapper on the model.
    Model has to be created and initialized before and fed here.
    Loss function, optimizer and scheduler are created here automatically based on the fed config.

    Arguments:
            cfg:    Training config read from the .yaml
            model:  Initalized model
    """

    def __init__(
        self,
        cfg: Union[dict, EasyDict],
        model: nn.Module,
        # loss,
        # optimizer_constructor,
        # scheduler,
        # metrics,
    ):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.loss = self._get_loss()
        self.num_target_labels = len(model.num_target_classes)
        self.metrics = self._get_metrics()

        # self.bbox_sizes_df = pd.DataFrame(columns=['ID','LF']+['shape{}'.format(i+1) for i in range(3)]).set_index('ID') # pd.read_csv('/home/cerdur/aurora-survival/logs/example_experiment/bbox_sizes.csv',index_col='ID') #

    def _get_metrics(self) -> nn.ModuleDict:
        metric_cfg = self.cfg.metrics
        metric_dict = nn.ModuleDict()
        for i in range(len(metric_cfg)):
            metric_dict[f"target:{i}"] = nn.ModuleDict({name: getattr(metric, name)(**opts) for name, opts in metric_cfg[f"target:{i}"].items()})
        return metric_dict

    def _get_loss(self) -> nn.Module:
        loss_cfg = self.cfg.training.loss
        if not hasattr(loss_cfg, "loss_weights"):
            loss_cfg.loss_weights = None
        elif isinstance(loss_cfg.loss_weights, int) or isinstance(loss_cfg.loss_weights, float):
            loss_cfg.loss_weights = list(loss_cfg.loss_weights)
        return CombinedLoss(
            self.cfg, 
            loss_instances=loss_cfg.loss_instances, 
            loss_weights=loss_cfg.loss_weights,
            repeat_inputs=loss_cfg.repeat_inputs)

    def configure_optimizers(self) -> dict:
        """Configure the optimizer for your classifier."""
        optimizer_name = list(self.cfg.training.optimizer.keys())[0]  # read the name from config
        optimizer = getattr(optim, optimizer_name)  # get the class from the name
        # self.optimizer_constructor = lambda x: optimizer(x, **self.cfg.optimizer[optimizer_name]) #

        self.optimizer = optimizer(self.parameters(), **self.cfg.training.optimizer[optimizer_name])
        # optimizer_dict = {"optimizer": self.optimizer, **dict(islice(self.cfg.optimizer.items(),1,100))} # if any addititional parameters given (e.g. frequency), add them -- DOESN'T WORK, check if ever necessary
        if self.cfg.training.scheduler.initializer is not None:
            scheduler_name = list(self.cfg.training.scheduler.initializer.keys())[0]
            scheduler = getattr(optim.lr_scheduler, scheduler_name)
            self.scheduler_constructor = lambda x: scheduler(x, **self.cfg.training.scheduler.initializer[scheduler_name])
            self.scheduler = self.scheduler_constructor(self.optimizer)
            scheduler_dict = {
                "scheduler": self.scheduler,
                **dict(islice(self.cfg.training.scheduler.items(), 1, 100)), # for monitor and frequency params
            }
            return {"optimizer": self.optimizer, "lr_scheduler": scheduler_dict}
        else:
            self.scheduler_constructor = None
            return self.optimizer

    def forward(self, img, tabular=None, **kwargs):
        return self.model(img, tabular, **kwargs)

    # def on_train_start(self) -> None:
    #     ## torchmetrics want to be on the same device as the model
    #     for i, label in enumerate(self.metrics):
    #         for name, func in self.metrics[label].items():
    #             try:
    #                 func.to(self.device)
    #             except:
    #                 pass
    # def on_test_start(self) -> None:
    #     ## torchmetrics want to be on the same device as the model
    #     for i, label in enumerate(self.metrics):
    #         for name, func in self.metrics[label].items():
    #             try:
    #                 func.to(self.device)
    #             except:
    #                 pass
    
    # def on_validation_start(self) -> None:
    #     ## torchmetrics want to be on the same device as the model
    #     for i, label in enumerate(self.metrics):
    #         for name, func in self.metrics[label].items():
    #             try:
    #                 func.to(self.device)
    #             except:
    #                 pass

    def on_train_epoch_start(self) -> None:
        
        if not self.model.freeze_encoder:
            return
        elif self.model.unfreeze_epoch is not None:
            ## encoder parameters are already frozen in model init()
            if self.current_epoch == 0:
                print("Freezing encoder")
                self.frozen_params = []
                for name, param in self.model.named_parameters():
                    if not param.requires_grad:
                        self.frozen_params.append(name)
                # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1e30, gamma=1)  ## keeping lr constant

            if self.current_epoch == self.model.unfreeze_epoch:
                print("Unfreezing encoder")
                for name, param in self.model.named_parameters():
                    if name in self.frozen_params:
                        param.requires_grad = True
                # if self.scheduler_constructor:
                #     self.scheduler = self.scheduler_constructor(self.optimizer)
        else:
            ## do nothing if no unfreeze specified
            return

    def training_step(self, batch: dict, batch_idx: int) -> dict:

        preds = self(batch["images"], batch["tabular"])  # list
        targets = batch["labels"].unbind(dim=1)
        total_loss, partial_losses = self.loss(list(zip(preds, targets)))
        out_dict = {"loss": total_loss, "partial_losses": partial_losses}
        for i in range(self.num_target_labels):
            out_dict[f"preds:{i}"] = preds[i].detach()
            out_dict[f"labels:{i}"] = targets[i].detach()

        return out_dict

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        
        preds = self(batch["images"], batch["tabular"])
        targets = batch["labels"].unbind(dim=1)
        total_loss, partial_losses = self.loss(list(zip(preds, targets)))
        out_dict = {"loss": total_loss, "partial_losses": partial_losses}
        for i in range(self.num_target_labels):
            out_dict[f"preds:{i}"] = preds[i].detach()
            out_dict[f"labels:{i}"] = targets[i].detach()

        return out_dict

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        preds = self(batch["images"], batch["tabular"])
        targets = batch["labels"].unbind(dim=1)
        total_loss, partial_losses = self.loss(list(zip(preds, targets)))
        out_dict = {"loss": total_loss, "partial_losses": partial_losses}
        for i in range(self.num_target_labels):
            out_dict[f"preds:{i}"] = preds[i].detach()
            out_dict[f"labels:{i}"] = targets[i].detach()
        out_dict["patient_id"] = batch["patient_id"]

        return out_dict

    @staticmethod
    def gather_outputs(outputs: list[dict]):
        gathered_outputs = {}
        for key in outputs[0].keys():
            if key == "patient_id":
                gathered_outputs[key] = [outs for out in outputs for outs in out[key]]
            elif key == "partial_losses":
                # list of size <num_losses> of tensors with shape (<cohort_size>,)
                gathered_outputs[key] = cat([tensor(out[key]).unsqueeze(0) for out in outputs]).unbind(dim=1)
            elif not outputs[0][key].shape:
                gathered_outputs[key] = tensor([out[key] for out in outputs])
            else:
                gathered_outputs[key] = cat([out[key] for out in outputs])

            if isinstance(gathered_outputs[key], MetaTensor):
                gathered_outputs[key] = gathered_outputs[key].as_tensor()

        return gathered_outputs

    def training_epoch_end(self, outputs: list[dict]):
        outputs = TrainingBase.gather_outputs(outputs)
        self.log("epoch", int(self.current_epoch))
        self.log("loss/train", mean(outputs["loss"]))
        if len(self.loss.loss_names) > 1:   
            for i, name in enumerate(self.loss.loss_names):
                self.log(name + "/train", mean(outputs["partial_losses"][i]))
        
        for i, label in enumerate(self.metrics):
            for name, func in self.metrics[label].items():
                self.log(name.lower() + "/train", func(outputs[f"preds:{i}"], outputs[f"labels:{i}"]))


    def validation_epoch_end(self, outputs: list[dict]):
        if not self.trainer.sanity_checking:
            outputs = TrainingBase.gather_outputs(outputs)
            self.log("epoch", int(self.current_epoch))
            self.log("loss/val", mean(outputs["loss"]))
            if len(self.loss.loss_names) > 1:   
                for i, name in enumerate(self.loss.loss_names):
                    self.log(name + "/val", mean(outputs["partial_losses"][i]))
            
            for i, label in enumerate(self.metrics):
                for name, func in self.metrics[label].items():
                    self.log(name.lower() + "/val", func(outputs[f"preds:{i}"], outputs[f"labels:{i}"]))

    def test_epoch_end(self, outputs: list[dict]):
        outputs = TrainingBase.gather_outputs(outputs)
        self.log("epoch", int(self.current_epoch))
        self.log("loss/test", mean(outputs["loss"]))
        # for i, name in enumerate(self.loss.loss_names):
        #     self.log(name + "/test", mean(outputs["partial_losses"][i]))
        
        for i, label in enumerate(self.metrics):
            for name, func in self.metrics[label].items():
                self.log(name.lower() + "/test", func(outputs[f"preds:{i}"], outputs[f"labels:{i}"]))


        
class CoxTrainingBase(TrainingBase):
    def __init__(
        self,
        cfg: Union[dict, EasyDict],
        model: nn.Module,
    ):
        super().__init__(cfg, model)
        self.save_hyperparameters(ignore=["model"])
        self.image_embeddings = []
        self.tabular_data = []
        self.labels = []
        self.image_embeddings_val = []
        self.tabular_data_val = []
        self.labels_val = []
        # loss, optimizer setup done
        # self.metrics = self._get_metrics() # update -- now we have multitargets, using the original trainingBase nested storage

    # def _get_metrics(self) -> nn.ModuleDict:
    #     metric_cfg = self.cfg.metrics
    #     metric_dict = {}
    #     #for i in range(self.num_target_labels):
    #     metric_dict = {name: getattr(metric,name)(**opts) for name,opts in metric_cfg[f'target:{0}'].items()}
    #     return nn.ModuleDict(metric_dict)

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()

        # load best model stage to compute embeddings
        # add continue_ckpt if here
        
        if (self.current_epoch == self.cfg.training.start_second_stage_epoch):
            if not self.cfg.training.continue_ckpt: # weights already loaded in the beginning
                if self.cfg.training.start_second_stage_epoch == 0 and hasattr(self.cfg, "continue_path"):
                    ckpt_path = self.cfg.continue_path / (f"fold-{self.cfg.fold}" if self.cfg.fold is not None else "") / "checkpoints" / "best_model.ckpt"
                else: 
                    ckpt_path = self.trainer.checkpoint_callback.dirpath + "/best_model.ckpt"
                # if starting 2nd stage at epoch 0, you probably load weights beforehand, so skip this
                print("Starting 2nd training stage by loading best checkpoint")
                state_dict = load(ckpt_path)["state_dict"]
                state_dict =  {k[6:]: v for k, v in state_dict.items()}
                self.model.load_state_dict(state_dict)
                for g in self.optimizer.param_groups:
                    g['lr'] = 1e-4 #self.cfg.optimizer.AdamW.lr
                self.scheduler._reset()

            #TODO: freeze encoder after this?



    def _pair_preds_targets(self, preds: list, targets: list) -> list:
        """
        prepare the pred-target pairing
        assumption: first two target entries are LF,LC (ensured in dataloading)
        """
        return [[preds[0], *targets[:2]]] + list(zip(preds[1:], targets[2:]))

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        # 2-stage training
        if self.current_epoch == self.cfg.training.start_second_stage_epoch:
            batch_embeddings = self.model.get_embeddings(batch["images"],batch["tabular"])
            self.image_embeddings.append(batch_embeddings.detach())
            self.tabular_data.append(batch["tabular"])
            self.labels.append(batch["labels"])
            if batch_idx == (self.cfg.training.len_train_loader - 1):
                self.image_embeddings = cat(self.image_embeddings, dim=0)
                self.tabular_data = cat(self.tabular_data,dim=0)
                self.labels = cat(self.labels, dim=0)
        elif self.current_epoch > self.cfg.training.start_second_stage_epoch:
            batch["images"] = self.image_embeddings
            batch["tabular"] = self.tabular_data
            batch["labels"] = self.labels
        preds = self(batch["images"], batch["tabular"])
        targets = list(batch["labels"].unbind(dim=1))
        loss, partial_losses = self.loss(
            self._pair_preds_targets(preds, targets)
        )  # loss_input: nested list as [[pred[0], events, durations],...,[pred[1],volumes]] (... if additional targets comes )
        out_dict = {
            "loss": loss,
            "patient_id": batch["patient_id"],
            "labels": batch["labels"].detach(),
            "partial_losses": partial_losses,
        }
        for i in range(len(self.metrics)):
            out_dict[f"preds:{i}"] = preds[i].detach()

        return out_dict

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        if self.current_epoch == self.cfg.training.start_second_stage_epoch:
            batch_embeddings = self.model.get_embeddings(batch["images"],batch["tabular"])
            self.image_embeddings_val.append(batch_embeddings.detach())
            self.tabular_data_val.append(batch["tabular"])
            self.labels_val.append(batch["labels"])
            if batch_idx == (self.cfg.training.len_val_loader - 1):
                self.image_embeddings_val = cat(self.image_embeddings_val, dim=0)
                self.tabular_data_val = cat(self.tabular_data_val,dim=0)
                self.labels_val = cat(self.labels_val, dim=0)
        elif self.current_epoch > self.cfg.training.start_second_stage_epoch:
            batch["images"] = self.image_embeddings_val
            batch["tabular"] = self.tabular_data_val
            batch["labels"] = self.labels_val
        
        preds = self(batch["images"], batch["tabular"])
        targets = list(batch["labels"].unbind(dim=1))
        loss, partial_losses = self.loss(self._pair_preds_targets(preds, targets))  # loss_input: nested list as [[pred, events, durations]]
        out_dict = {
            "loss": loss,
            "patient_id": batch["patient_id"],
            "labels": batch["labels"].detach(),
            "partial_losses": partial_losses,
        }
        for i in range(len(self.metrics)):
            out_dict[f"preds:{i}"] = preds[i].detach()

        return out_dict

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        preds = self(batch["images"], batch["tabular"])
        targets = list(batch["labels"].unbind(dim=1))
        loss, partial_losses = self.loss(self._pair_preds_targets(preds, targets)) # loss_input: nested list as [[pred, events, durations]]
        out_dict = {
            "loss": loss,
            "patient_id": batch["patient_id"],
            "labels": batch["labels"].detach(),
            "partial_losses": partial_losses,
        }
        for i in range(len(self.metrics)):
            out_dict[f"preds:{i}"] = preds[i].detach()

        return out_dict

    def training_epoch_end(self, outputs: list[dict]):
        outputs = TrainingBase.gather_outputs(outputs)
        self.log("epoch", int(self.current_epoch))
        # self.log("loss/val", mean(outputs["loss"]))
        targets = outputs["labels"].unbind(dim=1)
        metric_inputs = self._pair_preds_targets([outputs[f"preds:{i}"] for i in range(len(self.metrics))], targets)
        
        for i, name in enumerate(self.loss.loss_names):
            self.log(name + "/train", mean(outputs["partial_losses"][i]))
        self.log("loss/train", self.loss(metric_inputs)[0])
        for i, label in enumerate(self.metrics):
            for name, func in self.metrics[label].items():
                self.log(name.lower() + "/train", func(*metric_inputs[i]))
    
    def validation_epoch_end(self, outputs: list[dict]):
        if not self.trainer.sanity_checking:
            outputs = TrainingBase.gather_outputs(outputs)
            self.log("epoch", int(self.current_epoch))
            targets = outputs["labels"].unbind(dim=1)
            metric_inputs = self._pair_preds_targets([outputs[f"preds:{i}"] for i in range(len(self.metrics))], targets)
            for i, name in enumerate(self.loss.loss_names):
                self.log(name + "/val", mean(outputs["partial_losses"][i]))
            self.log("loss/val", self.loss(metric_inputs)[0])
            for i, label in enumerate(self.metrics):
                for name, func in self.metrics[label].items():
                    self.log(name.lower() + "/val", func(*metric_inputs[i]))
                    

    def test_epoch_end(self, outputs: list[dict]):
        outputs = TrainingBase.gather_outputs(outputs)
        self.log("epoch", int(self.current_epoch))
        targets = outputs["labels"].unbind(dim=1)
        metric_inputs = self._pair_preds_targets([outputs[f"preds:{i}"] for i in range(len(self.metrics))], targets)
        
        for i, name in enumerate(self.loss.loss_names):
            self.log(name + "/test", mean(outputs["partial_losses"][i]))
        self.log("loss/test", self.loss(metric_inputs)[0])
        for i, label in enumerate(self.metrics):
            for name, func in self.metrics[label].items():
                self.log(name.lower() + "/test", func(*metric_inputs[i]))
        if self.logger:
            res = {
                'patient_id':outputs['patient_id'],
                'preds': outputs['preds:0'].cpu().tolist(),
                'LF': targets[0].cpu().tolist(),
                'LC': targets[1].cpu().tolist()}
            res = DataFrame.from_dict(res)
            res.to_csv(self.logger.save_dir + f"/test_preds-{self.current_epoch}.csv")
            
