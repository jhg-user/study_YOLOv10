from ultralytics import YOLO

# 모델 파일 경로
best_saved_model_path = '/home/hkj/yolov10pj/yolov10_human/trainresult/result_facedetect13/weights/best.pt'

# 모델 로드
model = YOLO(best_saved_model_path)

# 데이터셋 경로 설정
data_path = '/home/hkj/yolov10pj/yolov10_human/dataset/dataset_human/dataset.yaml'

# 모델 평가 함수
def evaluate(model, data_path):
    # 모든 이미지에 대해 평가를 수행
    results = model.val(data=data_path, save_json=True)
    return results

# 평가 실행
results = evaluate(model, data_path)

# 적절한 속성을 사용하여 메트릭 출력
# 전체 클래스에 대한 mAP@0.5 및 mAP@0.5:0.95 출력
print(f"Validation mAP@0.5 for all classes: {results.maps[0][0]}")
print(f"Validation mAP@0.5:0.95 for all classes: {results.maps[0][1]}")

# 각 클래스별 메트릭 출력
for i, class_name in enumerate(results.names):
    print(f"Class {class_name}: mAP@0.5 = {results.maps[i][0]}, mAP@0.5:0.95 = {results.maps[i][1]}")

# 추가적인 메트릭 출력
print(f"Validation Box Loss for all classes: {results.box.loss}")
print(f"Validation Objectness Loss for all classes: {results.obj.loss}")
print(f"Validation Classification Loss for all classes: {results.cls.loss}")

