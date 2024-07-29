from ultralytics import YOLOv10
import os
import matplotlib.pyplot as plt

m_file = 'yolov10m.pt'

# 모델 불러오기
model = YOLOv10.from_pretrained('jameslahm/yolov10m')
model = YOLOv10(m_file)


# 학습 및 모델 저장 경로 설정
results = model.train(
    data="/home/hkj/yolov10pj/yolov10/dataset/WIDER_DARKNET/dataset.yaml",
    model=m_file, # model 지정
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

