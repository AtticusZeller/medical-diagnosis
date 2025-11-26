import numpy as np
import torch
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from rich import print
from timm.data.config import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch import nn
from torch.utils.data import DataLoader

from expt.data.dataset import DataModule


def create_rich_progress_bar() -> RichProgressBar:
    """Create a RichProgressBar instance.
    Ref: https://lightning.ai/docs/pytorch/stable/common/progress_bar.html"""
    return RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
            metrics_text_delimiter="\n",
            metrics_format=".3e",
        )
    )


def mean_std(data_module: DataModule) -> tuple[float, float]:
    dataset = data_module.dataset_class(
        data_module.data_dir, train=True, transform=data_module.transform
    )
    loader: DataLoader = DataLoader(dataset, batch_size=len(dataset), pin_memory=True)

    data = next(iter(loader))[0]
    return data.mean().item(), data.std().item()


def check_transform(model: nn.Module) -> None:
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    print(config)
    print(transform)


def to_tensor(
    data, requires_grad=True, dtype=torch.float32, device="cuda"
) -> torch.Tensor:
    """
    Convert numpy array or list to a PyTorch tensor.
    """
    if (
        not isinstance(data, list)
        and not isinstance(data, np.ndarray)
        and not torch.is_tensor(data)
        and not isinstance(data, int)
    ):
        raise TypeError("to tensor needs list,np.ndarray or tensor")

    data = torch.as_tensor(data, dtype=dtype, device=device)  # More efficient
    return data.requires_grad_(requires_grad)
