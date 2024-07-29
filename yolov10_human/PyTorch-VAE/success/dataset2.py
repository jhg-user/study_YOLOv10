import os
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from pytorch_lightning import LightningDataModule
from pathlib import Path
from typing import Optional, Union, List, Sequence
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = sorted(self.data_dir.glob('*.jpg'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = default_loader(img_path)

        if self.transform:
            img = self.transform(img)

        return img

class VAEDataset(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        image_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # std 값 추가
        ])

        val_transforms = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # std 값 추가
        ])

        self.train_dataset = MyDataset(
            data_dir=os.path.join(self.data_dir, 'train/human-face'),
            transform=train_transforms
        )

        self.val_dataset = MyDataset(
            data_dir=os.path.join(self.data_dir, 'val/non-human-face'),
            transform=val_transforms
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

