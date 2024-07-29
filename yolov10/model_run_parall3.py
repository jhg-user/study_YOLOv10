import os
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from ultralytics import YOLOv10
from ultralytics.utils import callbacks

parser = argparse.ArgumentParser(description='YOLOv10 Training Script')
parser.add_argument('--data', type=str, help='Path to dataset YAML file')
parser.add_argument('--project', type=str, default='/home/hkj/yolov10pj/yolov10/trainresult', help='Project directory where logs and models will be saved')
parser.add_argument('--name', type=str, default='result_facedetect', help='Name of the training run')
parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--img-size', type=int, default=640, help='Input image size')
parser.add_argument('--workers', type=int, default=4, help='Number of workers for data loading')
args = parser.parse_args()

# 모델 정의 및 초기화
model = YOLOv10.from_pretrained('jameslahm/yolov10s')
model = YOLOv10('yolov10s.pt')

# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DataParallel로 모델 감싸기
model = nn.DataParallel(model)

# 모델을 GPU로 옮기기
model = model.to(device)

# TensorBoard 로그 설정
log_dir = os.path.join(args.project, 'tensorboard_logs')
writer = SummaryWriter(log_dir=log_dir)

# 학습 및 모델 저장 경로 설정
results = model.train(
    batch_size=args.batch_size,
    img_size=args.img_size,
    workers=args.workers,
    device=device,
    project=args.project,
    name=args.name,
    callbacks=callbacks
)

# 결과 기록
for epoch, metrics in enumerate(results.metrics):
    writer.add_scalar('Loss/train', metrics['loss'], epoch)
    writer.add_scalar('Accuracy/train', metrics['accuracy'], epoch)
    writer.add_scalar('mAP/train', metrics['map'], epoch)

writer.close()

