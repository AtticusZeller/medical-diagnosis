from collections.abc import Callable
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2

from expt.utils import to_tensor


class ChestXray(VisionDataset):
    """Chest X-ray Pneumonia Dataset.

    Dataset structure:
    chest_xray/
        train/
            NORMAL/
            PNEUMONIA/
        test/
            NORMAL/
            PNEUMONIA/

    Note: The val folder is ignored. Validation split is created from training data.
    """

    classes = ["NORMAL", "PNEUMONIA"]

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        """
        Initialize ChestXray dataset.

        Parameters
        ----------
        root : str | Path
            Root directory containing chest_xray folder
        train : bool
            If True, load train set; if False, load test set
        transform : Callable | None
            Transform to apply to images
        target_transform : Callable | None
            Transform to apply to targets
        """
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root: Path = Path(self.root)
        self.train = train

        # Dataset path: root/chest_xray/[train|test]/[NORMAL|PNEUMONIA]
        split = "train" if train else "test"
        self.data_dir = self.root / "chest_xray" / split

        if not self.data_dir.exists():
            raise RuntimeError(
                f"Dataset not found at {self.data_dir}. "
                "Please ensure the dataset has been downloaded."
            )

        # Load all image paths and labels
        self.samples: list[tuple[Path, int]] = []
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue

            class_idx = self.class_to_idx[class_name]
            for img_path in class_dir.glob("*.jpeg"):
                self.samples.append((img_path, class_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {self.data_dir}")

    def __getitem__(self, index: int) -> tuple[Any | Tensor, int]:
        """
        Get image and label.

        Parameters
        ----------
        index : int
            Index of sample

        Returns
        -------
        tuple[Any, int]
            Image (after transform) and label
        """
        img_path, target = self.samples[index]

        # Load image as RGB
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.samples)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


class DataModule(L.LightningDataModule):
    """Lightning DataModule for dataset_class.

    Uses random_split to create train/val split from training data.
    Ref: `https://lightning.ai/docs/pytorch/stable/data/datamodule.html#lightningdatamodule`
    """

    # Dataset registry mapping names to dataset classes
    DATASET_REGISTRY = {"chest_xray": ChestXray}

    def __init__(
        self,
        dataset_name: str = "chest_xray",
        data_dir: str | Path = "./datasets",
        batch_size: int = 32,
        transforms: v2.Compose | None = None,
        val_split: float = 0.2,
    ) -> None:
        """
        Initialize DataModule.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to load
        data_dir : str | Path
            Root directory containing the dataset folders
        batch_size : int
            Batch size for dataloaders
        transforms : v2.Compose | None
            Transforms to apply to the data
        val_split : float
            Fraction of training data to use for validation
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.transform = transforms
        self.val_split = val_split
        self.num_workers = 8

        # Determine dataset class from directory structure
        self.dataset_name = dataset_name.lower()
        self.dataset_class = self.DATASET_REGISTRY[self.dataset_name]

        self.train_data: Subset[Any] | None = None
        self.val_data: Subset[Any] | None = None
        self.test_data: VisionDataset | None = None
        self.train_sampler: WeightedRandomSampler | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            # Load full training dataset
            full_data = self.dataset_class(
                self.data_dir, train=True, transform=self.transform
            )

            # # Split into train and val it works as classes are balanced
            # train_size = int(len(full_data) * (1 - self.val_split))
            # val_size = len(full_data) - train_size
            # self.train_data, self.val_data = random_split(
            #     full_data, [train_size, val_size]
            # )

            # Stratified split into train and val
            labels = [label for _, label in full_data.samples]
            train_idx, val_idx = train_test_split(
                np.arange(len(labels)),
                test_size=self.val_split,
                shuffle=True,
                stratify=labels,
                random_state=42,
            )
            self.train_data = Subset(full_data, train_idx)
            self.val_data = Subset(full_data, val_idx)

            # batch for the classes are not balanced
            train_labels = [labels[i] for i in train_idx]
            class_counts = torch.bincount(
                to_tensor(train_labels, dtype=torch.int64, requires_grad=False)
            )
            class_weights = 1.0 / class_counts.float()

            # Assign weights to each sample
            sample_weights = [
                float(class_weights[label].item()) for label in train_labels
            ]
            self.train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )

        if stage == "test" or stage is None:
            # Load test dataset
            self.test_data = self.dataset_class(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self) -> DataLoader:
        """This is the dataloader that the Trainer fit() method uses."""
        assert self.train_data is not None, (
            "Train data not initialized. Did you forget to call setup('fit')?"
        )
        assert self.train_sampler is not None, (
            "Train sampler not initialized. Did you forget to call setup('fit')?"
        )

        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,  #  for the classes are not balanced
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        """This is the dataloader that the Trainer fit() and validate() methods uses."""
        assert self.val_data is not None, (
            "Validation data not initialized. Did you forget to call setup('fit')?"
        )
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        """This is the dataloader that the Trainer test() method uses."""
        assert self.test_data is not None, (
            "Test data not initialized. Did you forget to call setup('test')?"
        )
        return DataLoader(
            self.test_data, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def prepare_data(self) -> None:
        """
        Prepare data (download, tokenize, etc.).

        This method is called only on one GPU in distributed training.
        For this dataset, we just verify it exists.
        """
        dataset_path = self.data_dir / self.dataset_name
        if not dataset_path.exists():
            raise RuntimeError(
                f"Dataset directory not found: {dataset_path}\n"
                "Please ensure the dataset has been downloaded."
            )
