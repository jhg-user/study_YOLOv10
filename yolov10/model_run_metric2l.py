from ultralytics import YOLOv10
import os
import matplotlib.pyplot as plt

m_file = 'yolov10l.pt'

# 모델 불러오기
model = YOLOv10.from_pretrained('jameslahm/yolov10l')
model = YOLOv10(m_file)


#last_saved_model_path = '/home/hkj/yolov10pj/yolov10/trainresult/result_facedetect11/weights/last.pt'

# 모델 로드
#model = YOLOv10(last_saved_model_path)

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
    device=2, # gpu
    # save_period=1,  # 1 epoch마다 모델 저장
    project='/home/hkj/yolov10pj/yolov10/trainresult',  # model 저장
    name='result_facedetect'
)

