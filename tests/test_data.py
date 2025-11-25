"""Tests for ChestXray dataset and DataModule."""

import pytest
import torch
from PIL import Image
from torchvision.transforms import v2

from expt.data.dataset import ChestXray, DataModule
from expt.data.transform import base_transform


class TestChestXray:
    """Tests for ChestXray dataset class."""

    @pytest.fixture
    def mock_dataset_structure(self, tmp_path):
        """Create a mock dataset directory structure with sample images."""
        # Create directory structure
        dataset_root = tmp_path / "chest_xray"
        for split in ["train", "test"]:
            for class_name in ["NORMAL", "PNEUMONIA"]:
                class_dir = dataset_root / split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)

                # Create dummy images
                num_images = 3 if split == "train" else 2
                for i in range(num_images):
                    img = Image.new("RGB", (100, 100), color=(i * 50, i * 50, i * 50))
                    img.save(class_dir / f"image_{i}.jpeg")

        return tmp_path

    def test_dataset_initialization(self, mock_dataset_structure):
        """Test that dataset initializes correctly."""
        dataset = ChestXray(root=mock_dataset_structure, train=True)

        assert len(dataset) == 6  # 3 images per class × 2 classes
        assert dataset.classes == ["NORMAL", "PNEUMONIA"]
        assert dataset.class_to_idx == {"NORMAL": 0, "PNEUMONIA": 1}

    def test_dataset_train_test_split(self, mock_dataset_structure):
        """Test that train and test splits load different data."""
        train_dataset = ChestXray(root=mock_dataset_structure, train=True)
        test_dataset = ChestXray(root=mock_dataset_structure, train=False)

        assert len(train_dataset) == 6  # 3 per class
        assert len(test_dataset) == 4  # 2 per class

    def test_dataset_missing_directory(self, tmp_path):
        """Test that appropriate error is raised when dataset not found."""
        with pytest.raises(RuntimeError, match="Dataset not found"):
            ChestXray(root=tmp_path, train=True)

    def test_dataset_getitem(self, mock_dataset_structure):
        """Test that __getitem__ returns correct format."""
        dataset = ChestXray(root=mock_dataset_structure, train=True)
        img, label = dataset[0]

        assert isinstance(img, Image.Image)
        assert isinstance(label, int)
        assert label in [0, 1]

    def test_dataset_with_transform(self, mock_dataset_structure):
        """Test that transforms are applied correctly."""
        transform = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Resize((224, 224))]
        )

        dataset = ChestXray(
            root=mock_dataset_structure, train=True, transform=transform
        )

        img, label = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 224, 224)
        assert img.dtype == torch.float32

    def test_dataset_empty_directory(self, tmp_path):
        """Test error handling when no images found."""
        # Create structure but no images
        dataset_root = tmp_path / "chest_xray" / "train"
        dataset_root.mkdir(parents=True)
        (dataset_root / "NORMAL").mkdir()
        (dataset_root / "PNEUMONIA").mkdir()

        with pytest.raises(RuntimeError, match="No images found"):
            ChestXray(root=tmp_path, train=True)


class TestDataModule:
    """Tests for DataModule class."""

    @pytest.fixture
    def mock_dataset_structure(self, tmp_path):
        """Create a mock dataset directory structure."""
        dataset_root = tmp_path / "chest_xray"
        for split in ["train", "test"]:
            for class_name in ["NORMAL", "PNEUMONIA"]:
                class_dir = dataset_root / split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)

                # Create more images for meaningful splits
                num_images = 10 if split == "train" else 5
                for i in range(num_images):
                    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
                    img.save(class_dir / f"image_{i}.jpeg")

        return tmp_path

    def test_datamodule_initialization(self, mock_dataset_structure):
        """Test DataModule initialization."""
        dm = DataModule(
            dataset_name="chest_xray", data_dir=mock_dataset_structure, batch_size=4
        )

        assert dm.dataset_name == "chest_xray"
        assert dm.batch_size == 4
        assert dm.val_split == 0.2

    def test_datamodule_setup_fit(self, mock_dataset_structure):
        """Test DataModule setup for training."""
        dm = DataModule(
            dataset_name="chest_xray",
            data_dir=mock_dataset_structure,
            batch_size=4,
            val_split=0.2,
        )

        dm.setup(stage="fit")

        # Total train images: 10 per class × 2 classes = 20
        # Val split: 20 * 0.2 = 4
        assert len(dm.train_data) == 16
        assert len(dm.val_data) == 4

    def test_datamodule_setup_test(self, mock_dataset_structure):
        """Test DataModule setup for testing."""
        dm = DataModule(
            dataset_name="chest_xray", data_dir=mock_dataset_structure, batch_size=4
        )

        dm.setup(stage="test")

        # Total test images: 5 per class × 2 classes = 10
        assert len(dm.test_data) == 10

    def test_datamodule_with_transforms(self, mock_dataset_structure):
        """Test DataModule with custom transforms."""
        dm = DataModule(
            dataset_name="chest_xray",
            data_dir=mock_dataset_structure,
            batch_size=2,
            transforms=base_transform(),
        )

        dm.setup(stage="fit")
        train_loader = dm.train_dataloader()

        imgs, labels = next(iter(train_loader))
        assert imgs.shape[1:] == (3, 224, 224)
        assert imgs.dtype == torch.float32

    def test_datamodule_prepare_data_missing(self, tmp_path):
        """Test prepare_data raises error when dataset missing."""
        dm = DataModule(dataset_name="chest_xray", data_dir=tmp_path, batch_size=4)

        with pytest.raises(RuntimeError, match="Dataset directory not found"):
            dm.prepare_data()

    def test_datamodule_invalid_dataset_name(self, mock_dataset_structure):
        """Test error handling for invalid dataset name."""
        with pytest.raises(KeyError):
            DataModule(
                dataset_name="invalid_dataset",
                data_dir=mock_dataset_structure,
                batch_size=4,
            )

    def test_datamodule_custom_val_split(self, mock_dataset_structure):
        """Test custom validation split ratio."""
        dm = DataModule(
            dataset_name="chest_xray",
            data_dir=mock_dataset_structure,
            batch_size=4,
            val_split=0.3,
        )

        dm.setup(stage="fit")

        # Total: 20 images, 30% validation
        assert len(dm.train_data) == 14
        assert len(dm.val_data) == 6
