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

class VAEDataset(Dataset):
    def __init__(self, data_path, image_size, batch_size, train_transforms=None, val_transforms=None, mode='train'):
        self.data_path = Path(data_path)
        self.image_size = image_size
        self.batch_size = batch_size
        self.mode = mode
        self.transforms = train_transforms if mode == 'train' else val_transforms
        
        self.image_paths = list(self.data_path.glob(f'{mode}/*/*'))
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transforms:
            for transform in self.transforms:
                transform_name, transform_params = transform.split(':', 1)
                transform_params = transform_params.strip('()').split(',')
                transform_params = tuple(map(int, transform_params))
                
                if transform_name == "Resize":
                    image = transforms.Resize(transform_params)(image)
                elif transform_name == "ToTensor":
                    image = transforms.ToTensor()(image)
                elif transform_name == "Normalize":
                    mean = float(transform_params[0])
                    std = float(transform_params[1])
                    image = transforms.Normalize((mean,), (std,))(image)
        
        return image

'''
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
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        val_transforms = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
'''
