from typing import Literal

from torchvision.transforms import v2 as v2

from .dataset import DataModule
from .transform import base_transform, efficientnetv2_pt_transform, resnet_pt_transform

__all__ = ["create_data_module", "DataModule"]


def create_data_module(
    name: str = "chest_xray",
    data_dir: str = "./datasets",
    batch_size: int = 32,
    transform: Literal["base", "resnet_pt", "efficientnetv2_pt"] = "base",
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
    # Get dataset class to access mean/std for standardization
    if transform == "base":
        return DataModule(
            dataset_name=name,
            data_dir=data_dir,
            batch_size=batch_size,
            transforms=base_transform(),
        )
    elif transform == "resnet_pt":
        return DataModule(
            dataset_name=name,
            data_dir=data_dir,
            batch_size=batch_size,
            transforms=resnet_pt_transform(),
        )
    elif transform == "efficientnetv2_pt":
        return DataModule(
            dataset_name=name,
            data_dir=data_dir,
            batch_size=batch_size,
            transforms=efficientnetv2_pt_transform(),
        )
    else:
        raise ValueError(f"Invalid transform type: {transform}")
