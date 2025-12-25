from pathlib import Path
from typing import Literal

import lightning.pytorch as pl
import timm
import torch
from rich import print
from torch import Tensor, is_tensor, nn
from torch.nn import CrossEntropyLoss, functional as F
from torch.optim import AdamW, Optimizer
from torchmetrics.functional import accuracy, auroc, f1_score, precision_recall_curve

import wandb
from expt import loss
from expt.config import Config
from expt.utils import check_transform


class BaseModel(pl.LightningModule):
    """wandb logger accumulate metrics and calculate ave for all steps"""

    loss: nn.Module
    lr: float

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.best_val_metric = 0.0
        self.best_threshold = 0.5

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """needs to return a loss from a single batch"""

        x, y = batch
        logits = self(x)
        # threshold at 0.5 for monitor accuracy
        preds = torch.argmax(logits, dim=1)
        loss = self.loss(logits, y)

        acc = accuracy(preds, y, task="binary")
        # Log loss and acc
        # steps for train loss, set log_every_n_steps for trainer in config.yml
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        """used for logging metrics"""

        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)

        # Log loss and f1
        # val/test focus on epoch metrics, step metrics are meaningless
        self.log("val_loss", loss, on_step=False, on_epoch=True)

        # accumulate outputs for epoch-end metrics
        self.validation_step_outputs.append(
            {"logits": logits.detach(), "labels": y.detach()}
        )

    def on_validation_epoch_end(self) -> None:
        """
        - calculate AUROC for all validation set
        - find the best threshold based on F1 score
        - log PR curve to wandb
        """
        if len(self.validation_step_outputs) == 0:
            return

        all_logits = torch.cat([x["logits"] for x in self.validation_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.validation_step_outputs])

        probs = F.softmax(all_logits, dim=1)
        probs_pos = probs[:, 1]

        # calculate AUROC
        aur = auroc(probs_pos, all_labels, task="binary")
        assert aur is not None, "AUROC calculation failed"
        self.log("val_auroc", aur, on_step=False, on_epoch=True)

        # log PR curve to wandb if AUROC improved
        if aur > self.best_val_metric:
            self.best_val_metric = aur
            wandb.log(
                {
                    "val_PR_curve": wandb.plot.pr_curve(
                        all_labels.cpu().numpy(),
                        probs.cpu().numpy(),
                        labels=["NORMAL", "PNEUMONIA"],
                        classes_to_plot=[1],  # PNEUMONIA class
                        title="Precision-Recall Curve on Validation Set",
                    )
                }
            )

        precisions, recalls, thresholds = precision_recall_curve(
            probs_pos, all_labels, task="binary"
        )
        assert is_tensor(precisions) and is_tensor(recalls), (
            "Precision or Recall calculation failed"
        )
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = torch.argmax(f1_scores)
        self.best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        self.log(
            "val_best_threshold", self.best_threshold, on_step=False, on_epoch=True
        )
        self.log("val_best_f1", best_f1, on_step=False, on_epoch=True)

        self.validation_step_outputs.clear()

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)

        # Log loss
        # val/test focus on epoch metrics, step metrics are meaningless
        self.log("test_loss", loss, on_step=False, on_epoch=True)

        # accumulate outputs for epoch-end metrics
        self.test_step_outputs.append({"logits": logits.detach(), "labels": y.detach()})

    def on_test_epoch_end(self) -> None:
        """
        - calculate AUROC for all validation set
        - calculate F1 score 、
        - calculate accuracy
        - log confusion matrix to wandb
        - log PR curve to wandb
        """
        if len(self.test_step_outputs) == 0:
            return

        all_logits = torch.cat([x["logits"] for x in self.test_step_outputs])
        all_labels = torch.cat([x["labels"] for x in self.test_step_outputs])

        probs = F.softmax(all_logits, dim=1)
        probs_pos = probs[:, 1]

        # calculate AUROC
        aur = auroc(probs_pos, all_labels, task="binary")
        assert aur is not None, "AUROC calculation failed"
        self.log("test_auroc", aur, on_step=False, on_epoch=True)

        # make predictions based on best threshold
        preds = (probs_pos >= self.best_threshold).long()

        # calculate F1 score
        test_f1 = f1_score(preds, all_labels, task="binary")
        self.log("test_f1", test_f1, on_step=False, on_epoch=True)

        # calculate accuracy
        test_acc = accuracy(preds, all_labels, task="binary")
        self.log("test_acc", test_acc, on_step=False, on_epoch=True)

        # Confusion matrix
        wandb.log(
            {
                "conf_mat": wandb.plot.confusion_matrix(
                    preds=preds.cpu().numpy().tolist(),
                    y_true=all_labels.cpu().numpy().tolist(),
                    class_names=["NORMAL", "PNEUMONIA"],
                )
            }
        )

        # PR curve
        wandb.log(
            {
                "test_PR_curve": wandb.plot.pr_curve(
                    all_labels.cpu().numpy(),
                    probs.cpu().numpy(),
                    labels=["NORMAL", "PNEUMONIA"],
                    classes_to_plot=[1],  # PNEUMONIA class
                    title="Precision-Recall Curve on Test Set",
                )
            }
        )
        self.test_step_outputs.clear()

    def configure_optimizers(self) -> Optimizer:
        """defines model optimizer"""
        return AdamW(self.parameters(), lr=self.lr)


class FineTuneBaseModel(BaseModel):
    model: nn.Module

    def freeze_except(self, trainable: list[str], debug: bool = False) -> None:
        """freeze all layers except the ones specified in trainable

        Parameters
        ----------
        trainable : list[str]
            list of layer names to be unfrozen
        debug : bool, optional
            print all finetune layers, by default False
        """
        # Freeze all layers
        for _, param in self.model.named_parameters():
            param.requires_grad = False
        # Unfreeze the layers
        fine_tune_params = []
        fine_tune_layers = set()
        for module_name, module in self.model.named_modules():
            if module_name in trainable:
                fine_tune_layers.add(module_name)
                for name, param in module.named_parameters():
                    param.requires_grad = True
                    fine_tune_params.append(module_name + "." + name)
        print(f"Fine-tuning {len(fine_tune_layers)} layers:")
        layers = list(fine_tune_layers)
        layers.sort()
        print(layers)
        if debug:
            print(f"Fine-tuning {len(fine_tune_params)} paras:")
            print(fine_tune_params)


class ResNet18Transfer(FineTuneBaseModel):
    """ResNet18 transfer learning model for Fashion MNIST
    Features:
    - Uses pretrained ResNet18 as backbone
    - Custom classification head
    - Supports feature extraction and fine-tuning
    """

    def __init__(
        self,
        num_classes: int = 10,
        lr: float = 1e-3,
        unfreeze_layers: list[str] | None = None,
        loss_fn: nn.Module | None = None,
    ) -> None:
        super().__init__()

        # Load ResNet18 model without pretrained weights
        self.model = timm.create_model(
            "resnet18", pretrained=True, num_classes=num_classes
        )
        # check_transform(self.model)
        # loss
        self.lr = lr
        self.loss = loss_fn if loss_fn is not None else CrossEntropyLoss()

        # save hyperparameters
        self.save_hyperparameters(ignore=["loss_fn"])
        if unfreeze_layers is not None:
            self.freeze_except(unfreeze_layers)
        # self.resnet = torch.compile(self.resnet)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class EfficientNetV2Transfer(FineTuneBaseModel):
    """EfficientNet transfer learning model for Fashion MNIST
    Features:
    - Uses pretrained EfficientNet as backbone
    - Custom classification head
    - Supports feature extraction and fine-tuning
    Ref: https://lightning.ai/docs/pytorch/stable/advanced/transfer_learning.html#example-imagenet-computer-vision
    """

    def __init__(
        self,
        num_classes: int = 10,
        efficient_version: Literal["s", "m", "l"] = "s",
        lr: float = 1e-3,
        unfreeze_layers: list[str] | None = None,
        loss_fn: nn.Module | None = None,
    ) -> None:
        super().__init__()

        self.model = timm.create_model(
            f"tf_efficientnetv2_{efficient_version}",
            pretrained=True,
            num_classes=num_classes,
        )
        check_transform(self.model)
        # loss
        self.lr = lr
        self.loss = loss_fn if loss_fn is not None else CrossEntropyLoss()

        # save hyperparameters
        self.save_hyperparameters()
        if unfreeze_layers is not None:
            self.freeze_except(unfreeze_layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class ViTTransfer(FineTuneBaseModel):
    """Vision Transformer transfer learning model

    Features:
    - Uses pretrained ViT as backbone (ImageNet-21k pretrained recommended)
    - Custom classification head (replace the original head)
    - Supports partial fine-tuning (only head, or head + last N blocks)

    Architecture:
    - Patch Embedding: 224×224 image → 196 patches (14×14) of 16×16 pixels
    - Transformer Encoder: Multi-head self-attention + MLP blocks
    - Classification Head: [CLS] token → Linear(hidden_dim, num_classes)
    """

    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = "vit_base_patch16_224",  # or vit_small_patch16_224
        lr: float = 1e-4,
        unfreeze_layers: list[str] | None = None,
        loss_fn: nn.Module | None = None,
    ) -> None:
        super().__init__()

        # Load pretrained ViT from timm
        # Available models: vit_tiny/small/base/large_patch16_224
        # Add '.augreg_in21k_ft_in1k' suffix for ImageNet-21k pretrained models
        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=num_classes
        )

        # Check model's expected input transformation
        check_transform(self.model)

        # Optimizer & Loss
        self.lr = lr
        self.loss = loss_fn if loss_fn is not None else CrossEntropyLoss()

        # Save hyperparameters for checkpoint
        self.save_hyperparameters(ignore=["loss_fn"])

        # Freeze layers if specified
        if unfreeze_layers is not None:
            self.freeze_except(unfreeze_layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class SwinV2Transfer(FineTuneBaseModel):
    """Swin Transformer V2 Transfer Learning Model

    Why SOTA?
    - Hierarchical structure (like CNNs) provides multi-scale features.
    - Shifted Windows approach enables linear complexity (efficiency).
    - V2 specific: Better scaling and stability via Post-Norm and Cosine Attention.

    Architecture:
    - Shifted Window Attention: Local attention within windows +
        shifting to cross windows.
    - Head: 'head' layer (Linear)
    """

    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = "swinv2_tiny_window8_256",
        lr: float = 1e-4,
        unfreeze_layers: list[str] | None = None,
        loss_fn: nn.Module | None = None,
    ) -> None:
        super().__init__()

        # Load SwinV2 from timm
        # Common choices:
        # - swinv2_tiny_window8_256 (Efficient, good for small data)
        # - swinv2_base_window12_192_22k (Pretrained on huge dataset, robust features)
        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=num_classes
        )

        check_transform(self.model)

        self.lr = lr
        self.loss = loss_fn if loss_fn is not None else CrossEntropyLoss()

        self.save_hyperparameters(ignore=["loss_fn"])

        if unfreeze_layers is not None:
            self.freeze_except(unfreeze_layers)
        else:
            # 默认只微调 Head
            self.freeze_except(["head"])

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class DenseNetTransfer(FineTuneBaseModel):
    """DenseNet transfer learning model

    Relevance:
    - Heavily used in medical imaging (e.g., CheXNet uses DenseNet121).
    - Features intensive feature reuse through dense connections.

    Architecture:
    - Backbone: DenseNet121/169/201
    - Head: 'classifier' layer (Linear)
    """

    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = "densenet121",  # Default to CheXNet backbone
        lr: float = 1e-3,
        unfreeze_layers: list[str] | None = None,
        loss_fn: nn.Module | None = None,
    ) -> None:
        super().__init__()

        # Load DenseNet from timm
        # Common choices: densenet121, densenet161, densenet169, densenet201
        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=num_classes
        )

        check_transform(self.model)

        self.lr = lr
        self.loss = loss_fn if loss_fn is not None else CrossEntropyLoss()

        self.save_hyperparameters(ignore=["loss_fn"])

        if unfreeze_layers is not None:
            self.freeze_except(unfreeze_layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


def create_model(config: Config, model_path: Path | None = None) -> BaseModel:
    """Factory function to create model based on config"""
    loss_fn = loss.create_loss_function(config.loss)
    model_name = config.model.name.lower()

    if "swin" in model_name:
        return (
            SwinV2Transfer(
                num_classes=config.data.num_classes,
                model_name=config.model.name,
                lr=config.optimizer.lr,
                unfreeze_layers=config.model.unfreeze_layers,
                loss_fn=loss_fn,
            )
            if model_path is None
            else SwinV2Transfer.load_from_checkpoint(model_path)
        )

    elif "vit" in model_name:
        return (
            ViTTransfer(
                num_classes=config.data.num_classes,
                model_name=config.model.name,
                lr=config.optimizer.lr,
                unfreeze_layers=config.model.unfreeze_layers,
                loss_fn=loss_fn,
            )
            if model_path is None
            else ViTTransfer.load_from_checkpoint(model_path)
        )
    elif "densenet" in model_name:
        return (
            DenseNetTransfer(
                num_classes=config.data.num_classes,
                model_name=config.model.name,
                lr=config.optimizer.lr,
                unfreeze_layers=config.model.unfreeze_layers,
                loss_fn=loss_fn,
            )
            if model_path is None
            else DenseNetTransfer.load_from_checkpoint(model_path)
        )

    elif model_name == "resnet18":
        return (
            ResNet18Transfer(
                num_classes=config.data.num_classes,
                lr=config.optimizer.lr,
                unfreeze_layers=config.model.unfreeze_layers,
                loss_fn=loss_fn,
            )
            if model_path is None
            else ResNet18Transfer.load_from_checkpoint(model_path)
        )
    elif model_name == "efficientnet_v2":
        return (
            EfficientNetV2Transfer(
                num_classes=config.data.num_classes,
                lr=config.optimizer.lr,
                efficient_version=config.model.efficient_version or "s",
                unfreeze_layers=config.model.unfreeze_layers,
                loss_fn=loss_fn,
            )
            if model_path is None
            else EfficientNetV2Transfer.load_from_checkpoint(model_path)
        )

    else:
        raise ValueError(f"Model name {config.model.name} not supported.")
