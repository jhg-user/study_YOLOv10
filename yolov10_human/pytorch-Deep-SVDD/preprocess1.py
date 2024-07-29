# dataset load

# we used the precomputed min_max values from the original implementation:
# https://github.com/lukasruff/Deep-SVDD-PyTorch/blob/1901612d595e23675fb75c4ebb563dd0ffebc21e/src/datasets/mnist.py

import os
import torch
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from PIL import Image

from utils.utils import global_contrast_normalization

## 폴더 이름을 클래스명으로
class CustomYOLODataset(data.Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        self.images = []
        self.labels = []
        self.class_to_label = {}

        # 폴더 이름을 클래스 레이블로 매핑
        for label_type in os.listdir(image_dir):
            label_path = os.path.join(image_dir, label_type)
            if os.path.isdir(label_path):
                self.class_to_label[label_type] = len(self.class_to_label)
                for image_file in os.listdir(label_path):
                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):  # 이미지 파일만 포함
                        image_path = os.path.join(label_path, image_file)
                        self.images.append(image_path)
                        self.labels.append(self.class_to_label[label_type])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label = self.labels[index]
        #image = Image.open(image_path).convert("L")  # 그레이스케일로 변환

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
        transforms.ToTensor(),
        transforms.Resize((28, 28), antialias=True),  # 크기 조정 필요 시
        transforms.Normalize((0.5,), (0.5,))  # 정규화
    ])

    train_image_dir = os.path.join(data_dir, 'train')
    print(f'load train data path : {train_image_dir}')
    val_image_dir = os.path.join(data_dir, 'val')
    print(f'load val data path : {val_image_dir}')

    train_dataset = CustomYOLODataset(train_image_dir, transform)
    val_dataset = CustomYOLODataset(val_image_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader
##
