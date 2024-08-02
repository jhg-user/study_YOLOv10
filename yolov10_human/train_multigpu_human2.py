import os
import matplotlib.pyplot as plt
import signal
import sys
import torch
from ultralytics import YOLOv10

def signal_handler(sig, frame):
    print('You pressed Ctrl+C! Exiting gracefully...')
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    #torch.cuda.empty_cache()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

m_file = 'yolov10l.pt'

# 모델 불러오기
model = YOLOv10.from_pretrained('jameslahm/yolov10l')
model = YOLOv10(m_file)

# 학습 재개
#m_file = '/home/hkj/yolov10pj/yolov10_human/trainresult/result_facedetect13/weights/last.pt'

# 모델 로드
model = YOLOv10(m_file)

# 학습 및 모델 저장 경로 설정
results = model.train(
    #data="/home/hkj/yolov10pj/yolov10_human/dataset/dataset_human/dataset.yaml",
    data="/home/hkj/yolov10pj/yolov10_human/dataset/dataset_human2/dataset.yaml",
    model=m_file, # model 지정
    #epochs=100,
    #epochs=30,
    epochs=30,
    batch=16,
    #batch=8,
    imgsz=640,
    workers=4,
    device=(1,2,3,4), # gpu
    # save_period=1,  # 1 epoch마다 모델 저장
    project='/home/hkj/yolov10pj/yolov10_human/trainresult',  # model 저장
    name='result_facedetect',
    #resume = True
)

# 학습 종료 후 메모리 비우기
torch.cuda.empty_cache()
