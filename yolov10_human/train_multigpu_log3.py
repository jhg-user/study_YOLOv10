import os
import signal
import sys
import torch
from ultralytics import YOLOv10
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import logging

def signal_handler(sig, frame):
    print('Ctrl+C를 눌렀습니다! 안전하게 종료합니다...')
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# 로그 설정
logging.basicConfig(filename='image_loading_errors.log', level=logging.ERROR)

class CustomDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.image_paths = self.get_image_paths(dataset_path)

    def get_image_paths(self, path):
        image_paths = []
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img
        except Exception as e:
            logging.error(f"이미지 로딩 오류 {img_path}: {e}")
            return torch.zeros((3, 640, 640))  # 빈 이미지 또는 0으로 채워진 텐서를 반환

transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# 학습 및 검증 데이터셋 경로 설정
train_dataset_path = "/home/hkj/yolov10pj/yolov10_human/dataset/dataset_human2/images/train"
val_dataset_path = "/home/hkj/yolov10pj/yolov10_human/dataset/dataset_human2/images/val"

# 경로 유효성 검사
if not os.path.exists(train_dataset_path):
    raise ValueError(f"학습 데이터셋 경로가 존재하지 않습니다: {train_dataset_path}")

if not os.path.exists(val_dataset_path):
    raise ValueError(f"검증 데이터셋 경로가 존재하지 않습니다: {val_dataset_path}")

# 데이터셋 및 데이터로더 생성
train_dataset = CustomDataset(train_dataset_path, transform=transform)
val_dataset = CustomDataset(val_dataset_path, transform=transform)

if len(train_dataset) == 0:
    raise ValueError("학습 데이터셋에 이미지가 없습니다.")
if len(val_dataset) == 0:
    raise ValueError("검증 데이터셋에 이미지가 없습니다.")

def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return torch.zeros((1, 3, 640, 640))  # 기본 반환값 설정
    return torch.utils.data.dataloader.default_collate(batch)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=custom_collate)

# 모델 불러오기
m_file = 'yolov10l.pt'
model = YOLOv10.from_pretrained('jameslahm/yolov10l')
model = YOLOv10(m_file)

# 학습 및 모델 저장 경로 설정
results = model.train(
    data={'train': train_loader, 'val': val_loader},  # 학습 및 검증 데이터로더 지정
    model=m_file,  # 모델 지정
    epochs=30,
    batch=16,
    imgsz=640,
    workers=4,
    device=(1, 2, 3, 4),  # GPU
    project='/home/hkj/yolov10pj/yolov10_human/trainresult',  # 모델 저장 경로
    name='result_facedetect',
)

# 학습 종료 후 메모리 비우기
torch.cuda.empty_cache()

