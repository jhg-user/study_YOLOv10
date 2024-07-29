import os
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from pytorch_lightning import LightningDataModule
from pathlib import Path
from typing import Optional, Union, List, Sequence
from PIL import Image

class VAEDataset(Dataset):
    def __init__(self, data_path, image_size, batch_size, mode='train'):
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.batch_size = batch_size
        self.mode = mode

        self.image_paths = list(self.data_path.glob(f'{mode}/*/*'))

        self.transforms = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transforms:
            image = self.transforms(image)

        return image

class VAEDataModule(LightningDataModule):
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

        self.data_path = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = VAEDataset(
            data_path=self.data_path,
            image_size=self.image_size,
            batch_size=self.train_batch_size,
            mode='train'
        )

        self.val_dataset = VAEDataset(
            data_path=self.data_path,
            image_size=self.image_size,
            batch_size=self.val_batch_size,
            mode='val'
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

