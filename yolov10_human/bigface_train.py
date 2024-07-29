from ultralytics import YOLOv10

# 기존 모델 로드
# wider face 학습된 yolov10s.pt
pretrained_model_path = '/home/hkj/yolov10pj/yolov10_human/trainresult/result_facedetect3/weights/best.pt'
model = YOLOv10(pretrained_model_path)

# 추가 학습 데이터셋 설정 파일 경로
additional_dataset_path = '/home/hkj/yolov10pj/yolov10_human/dataset/bigface_h/dataset.yaml'

# 전이 학습 수행
model.train(
    data=additional_dataset_path,
    epochs=30,
    imgsz=640,
    batch=8,
    device=2,
    model=pretrained_model_path,
    save=True
)

