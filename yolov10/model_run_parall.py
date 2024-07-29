from ultralytics import YOLOv10
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from ultralytics.utils import callbacks

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
log_dir = '/home/hkj/yolov10pj/yolov10/trainresult/tensorboard_logs'
writer = SummaryWriter(log_dir=log_dir)

# 학습 및 모델 저장 경로 설정
results = model.train(
    data="/home/hkj/yolov10pj/yolov10/dataset/WIDER_DARKNET/dataset.yaml",
    model="yolov10s.pt",  # model 지정
    batch=8,
    epochs=100,
    imgsz=640,
    workers=4,
    device=device,  # gpu
    project='/home/hkj/yolov10pj/yolov10/trainresult',  # model 저장
    name='result_facedetect',
    callbacks=callbacks  # 콜백 함수 설정
)

'''
# 결과 기록
for epoch, metrics in enumerate(results.metrics):
    writer.add_scalar('Loss/train', metrics['loss'], epoch)
    writer.add_scalar('Accuracy/train', metrics['accuracy'], epoch)
    writer.add_scalar('mAP/train', metrics['map'], epoch)

writer.close()
'''
