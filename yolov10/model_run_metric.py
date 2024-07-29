from ultralytics import YOLOv10
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


# 모델 불러오기
model = YOLOv10.from_pretrained('jameslahm/yolov10s')
model = YOLOv10('yolov10s.pt')


# TensorBoard 로그 설정
writer = SummaryWriter(log_dir='/home/hkj/yolov10pj/yolov10/trainresult/tensorboard_logs')

# 학습 및 모델 저장 경로 설정
results = model.train(
    data="/home/hkj/yolov10pj/yolov10/dataset/WIDER_DARKNET/dataset.yaml",
    model="yolov10s.pt", # model 지정
    #epochs=100,
    epochs=30,
    batch=4,
    #batch=8,
    imgsz=640,
    workers=4,
    device=0, # gpu
    # save_period=1,  # 1 epoch마다 모델 저장
    project='/home/hkj/yolov10pj/yolov10/trainresult',  # model 저장
    name='result_facedetect'
)

for epoch, metrics in enumerate(results.metrics):
    # TensorBoard에 로그 기록
    writer.add_scalar('Loss/train', metrics['loss'], epoch)
    writer.add_scalar('Accuracy/train', metrics['accuracy'], epoch)
    writer.add_scalar('mAP/train', metrics['map'], epoch)

writer.close()
