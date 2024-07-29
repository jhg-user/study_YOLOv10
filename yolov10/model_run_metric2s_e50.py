from ultralytics import YOLOv10
import os
import matplotlib.pyplot as plt

# 모델 불러오기
model = YOLOv10.from_pretrained('jameslahm/yolov10s')
model = YOLOv10('yolov10s.pt')


# 학습 및 모델 저장 경로 설정
results = model.train(
    data="/home/hkj/yolov10pj/yolov10/dataset/WIDER_DARKNET/dataset.yaml",
    model="yolov10s.pt", # model 지정
    #epochs=100,
    epochs=50,
    batch=8,
    #batch=8,
    imgsz=640,
    workers=4,
    device=1, # gpu
    # save_period=1,  # 1 epoch마다 모델 저장
    project='/home/hkj/yolov10pj/yolov10/trainresult',  # model 저장
    name='result_facedetect'
)

