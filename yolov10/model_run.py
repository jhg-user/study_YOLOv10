# 학습 모델 중간에 저장
from ultralytics import YOLOv10
import os
import matplotlib.pyplot as plt

model = YOLOv10.from_pretrained('jameslahm/yolov10s')

# 학습
#C:\Users\kii\yolov10\datasets\WIDER_DARKNET
model = YOLOv10('yolov10s.pt')

# 학습 및 모델 저장 경로 설정
results = model.train(
    data="/content/dataset/WIDER_DARKNET/dataset.yaml",
    model="yolov10s.pt", # model 지정
    batch=8,
    epochs=100,
    imgsz=640,
    workers=4,
    device=0, # gpu
    #save_period=1,  # 1 epoch마다 모델 저장
    project='/home/hkj/yolov10pj/yolov10/trainresult',  # model 저장
    name='result_facedetect'
)
