from collections.abc import Callable
from pathlib import Path
from typing import Any

import lightning as L
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2


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
    train_data: Subset[Any]
    val_data: Subset[Any]
    test_data: VisionDataset

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

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            # Load full training dataset
            full_data = self.dataset_class(
                self.data_dir, train=True, transform=self.transform
            )

            # Split into train and val
            train_size = int(len(full_data) * (1 - self.val_split))
            val_size = len(full_data) - train_size
            self.train_data, self.val_data = random_split(
                full_data, [train_size, val_size]
            )

        if stage == "test" or stage is None:
            # Load test dataset
            self.test_data = self.dataset_class(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self) -> DataLoader:
        """This is the dataloader that the Trainer fit() method uses."""
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        """This is the dataloader that the Trainer fit() and validate() methods uses."""
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        """This is the dataloader that the Trainer test() method uses."""
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
