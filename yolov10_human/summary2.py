from ultralytics import YOLO
import torch

# 모델 파일 경로
best_saved_model_path = '/home/hkj/yolov10pj/yolov10_human/runs/detect/train4/weights/best.pt'

# 모델 로드
model = YOLO(best_saved_model_path)

# 데이터셋 경로 설정
#data_path = '/home/hkj/yolov10pj/yolov10_human/dataset.yaml'
data_path = '/home/hkj/yolov10pj/yolov10_human/dataset/dataset_human/dataset.yaml'

# 모델 평가 함수
def evaluate(model, data_path):
    # 모든 이미지에 대해 평가를 수행
    results = model.val(data=data_path, save_json=True)
    return results

# 평가 실행
results = evaluate(model, data_path)


# 평가 결과 출력 (클래스 0 - human-face에 대해서만 출력)
metrics = results.metrics  # 최신 ultralytics 패키지에서는 metrics 속성에 저장됨
print(f"Validation mAP@0.5 for human-face: {metrics['metrics/mAP_0.5']}")
print(f"Validation mAP@0.5:0.95 for human-face: {metrics['metrics/mAP_0.5:0.95']}")
print(f"Validation Box Loss for human-face: {metrics['metrics/box_loss']}")
print(f"Validation Objectness Loss for human-face: {metrics['metrics/obj_loss']}")
print(f"Validation Classification Loss for human-face: {metrics['metrics/cls_loss']}")

