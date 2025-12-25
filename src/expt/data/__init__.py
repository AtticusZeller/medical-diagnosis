from typing import Literal

import torch
from torchvision.transforms import v2 as v2
from torchvision.transforms.transforms import InterpolationMode

from .dataset import DataModule

__all__ = ["create_data_module", "DataModule"]


def create_data_module(
    name: str = "chest_xray",
    data_dir: str = "./datasets",
    batch_size: int = 32,
    transform: Literal[
        "resnet_pt", "efficientnetv2_pt", "vit_pt", "swinv2_tiny_pt", "densnet_pt"
    ] = "resnet_pt",
) -> DataModule:
    """
    Create a DataModule for the specified dataset.

    Parameters
    ----------
    name : str
        Dataset name (currently supports "chest_xray")
    data_dir : str
        Root directory containing the dataset
    batch_size : int
        Batch size for dataloaders
    transform : Literal
        Transform type to apply

    Returns
    -------
    DataModule
        Configured DataModule instance
    """

    if transform == "resnet_pt":
        return DataModule(
            dataset_name=name,
            data_dir=data_dir,
            batch_size=batch_size,
            transforms=v2.Compose(
                [
                    v2.ToImage(),
                    v2.Resize(
                        235,
                        interpolation=InterpolationMode.BICUBIC,
                        max_size=None,
                        antialias=True,
                    ),
                    v2.CenterCrop(224),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
        )
    elif transform == "efficientnetv2_pt":
        return DataModule(
            dataset_name=name,
            data_dir=data_dir,
            batch_size=batch_size,
            transforms=v2.Compose(
                [
                    v2.ToImage(),
                    v2.Resize(
                        300,
                        interpolation=InterpolationMode.BICUBIC,
                        max_size=None,
                        antialias=True,
                    ),
                    v2.CenterCrop(300),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            ),
        )
    elif transform == "vit_pt":
        return DataModule(
            dataset_name=name,
            data_dir=data_dir,
            batch_size=batch_size,
            transforms=v2.Compose(
                [
                    v2.ToImage(),
                    v2.Resize(
                        248,
                        interpolation=InterpolationMode.BICUBIC,
                        max_size=None,
                        antialias=True,
                    ),
                    v2.CenterCrop(224),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(
                        mean=[0.5000, 0.5000, 0.5000], std=[0.5000, 0.5000, 0.5000]
                    ),
                ]
            ),
        )
    elif transform == "densnet_pt":
        return DataModule(
            dataset_name=name,
            data_dir=data_dir,
            batch_size=batch_size,
            transforms=v2.Compose(
                [
                    v2.ToImage(),
                    v2.Resize(
                        256,
                        interpolation=InterpolationMode.BICUBIC,
                        max_size=None,
                        antialias=True,
                    ),
                    v2.CenterCrop(224),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(
                        mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
                    ),
                ]
            ),
        )
    elif transform == "swinv2_tiny_pt":
        return DataModule(
            dataset_name=name,
            data_dir=data_dir,
            batch_size=batch_size,
            transforms=v2.Compose(
                [
                    v2.ToImage(),
                    v2.Resize(
                        284,
                        interpolation=InterpolationMode.BICUBIC,
                        max_size=None,
                        antialias=True,
                    ),
                    v2.CenterCrop(256),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(
                        mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
                    ),
                ]
            ),
        )
    elif transform == "swinv2_base_pt":
        return DataModule(
            dataset_name=name,
            data_dir=data_dir,
            batch_size=batch_size,
            transforms=v2.Compose(
                [
                    v2.ToImage(),
                    v2.Resize(
                        213,
                        interpolation=InterpolationMode.BICUBIC,
                        max_size=None,
                        antialias=True,
                    ),
                    v2.CenterCrop(192),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(
                        mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
                    ),
                ]
            ),
        )
    else:
        raise ValueError(f"Invalid transform type: {transform}")
