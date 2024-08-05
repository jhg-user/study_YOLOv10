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
        self.image_paths = [os.path.join(dataset_path, img) for img in os.listdir(dataset_path) if img.endswith(('.png', '.jpg', '.jpeg'))]

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

dataset_path = "/home/hkj/yolov10pj/yolov10_human/dataset/dataset_human2/images"

dataset = CustomDataset(dataset_path, transform=transform)
if len(dataset) == 0:
    raise ValueError("데이터셋에 이미지가 없습니다.")

def custom_collate(batch):
    # None 값을 필터링하여 유효한 데이터를 반환
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return torch.zeros((1, 3, 640, 640))  # 기본 반환값 설정
    return torch.utils.data.dataloader.default_collate(batch)

data_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=custom_collate)


m_file = 'yolov10l.pt'

# 모델 불러오기
model = YOLOv10.from_pretrained('jameslahm/yolov10l')
model = YOLOv10(m_file)

# 학습 및 모델 저장 경로 설정
results = model.train(
    data=data_loader,  # 커스텀 데이터셋 사용
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

