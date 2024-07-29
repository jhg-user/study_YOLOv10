import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.utils import global_contrast_normalization

class CustomYOLODataset(data.Dataset):
    def __init__(self, image_dir, normal_class='human-face', transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.normal_class = normal_class

        self.images = []
        self.labels = []

        # 폴더 이름을 클래스 레이블로 매핑
        for label_type in os.listdir(image_dir):
            label_path = os.path.join(image_dir, label_type)
            if os.path.isdir(label_path):
                label = 0 if label_type == self.normal_class else 1
                for image_file in os.listdir(label_path):
                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):  # 이미지 파일만 포함
                        image_path = os.path.join(label_path, image_file)
                        self.images.append(image_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label = self.labels[index]

        try:
            image = Image.open(image_path).convert("L")  # 그레이스케일로 변환
        except (OSError, IOError) as e:
            print(f"Error opening image {image_path}: {e}")
            image = Image.fromarray(np.zeros((224, 224), dtype=np.uint8))  # 빈 이미지

        if self.transform:
            image = self.transform(image)

        return image, label

def custom_yolo_loader(args, data_dir='./dataset/'):
    normal_class = 'human-face'

    transform = transforms.Compose([
        transforms.Resize((28, 28), antialias=True),  # 크기 조정 필요 시
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 정규화
    ])

    train_image_dir = os.path.join(data_dir, 'train')
    print(f'Load train data path: {train_image_dir}')
    val_image_dir = os.path.join(data_dir, 'val')
    print(f'Load val data path: {val_image_dir}')

    train_dataset = CustomYOLODataset(train_image_dir, normal_class=normal_class, transform=transform)
    val_dataset = CustomYOLODataset(val_image_dir, normal_class=normal_class, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

