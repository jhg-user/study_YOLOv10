import torch
import cv2
import numpy as np
from ultralytics import YOLOv10
from torchvision import transforms
from model import network
from PIL import Image
import torch.nn.functional as F

# YOLO 모델 로딩
def load_yolo_model(model_path):
    model = YOLOv10(model_path)
    return model

# Deep SVDD 모델 로딩
def load_svdd_model(model_path):
    model = torch.load(model_path)
    #model.eval()
    return model
def load_svdd_model(path):
    # 모델 객체 생성 (저장할 때와 같은 클래스 사용)
    model = network(z_dim=32)

    # 체크포인트 로드
    checkpoint = torch.load(path)

    # 모델의 state_dict 로드
    #model.load_state_dict(checkpoint[path])
        # 체크포인트에서 state_dict 추출
    state_dict = checkpoint['net_dict']
    #state_dict = checkpoint  # 만약 체크포인트가 직접 state_dict인 경우

    # 모델의 state_dict 로드
    model.load_state_dict(state_dict)

    # 평가 모드로 전환
    model.eval()

    return model

# YOLO 모델을 사용하여 이미지에서 객체를 탐지
def predict_yolo(model, image_path):
    #image = cv2.imread(image_path)
    #results = model(image)
    results = model.predict(image_path, save=True, imgsz=640, conf=0.5, device=0)
    #return results.boxes.xyxy[0]  # 결과 반환
    return results

'''
# 재구성 오차 계산을 위한 함수
def compute_reconstruction_error(data, model):
    with torch.no_grad():
        #data = torch.tensor(data, dtype=torch.float32)
        # `data` 텐서 추출
        data = data.cpu().numpy()  # GPU에서 CPU로 이동하고 NumPy 배열로 변환
        reconstructions = model(data)
        error = torch.mean((data - reconstructions) ** 2, dim=1)
        #return error.numpy()
        return error
'''
def compute_reconstruction_error(boxes, model, image_path):
    errors = []
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # 모델 입력 크기에 맞게 조정
        transforms.ToTensor(),
        transforms.Grayscale(),  # 모델이 그레이스케일 입력을 기대하는 경우
    ])

    for box in boxes:
        # box 좌표 추출
        xyxy = box.xyxy[0].tolist()
        left, top, right, bottom = map(int, xyxy[:4])

        # 이미지를 crop하고 Tensor로 변환
        cropped_image = image.crop((left, top, right, bottom))
        data = transform(cropped_image).unsqueeze(0)  # 배치 차원 추가

        # 모델을 사용하여 재구성
        reconstructions = model(data)
        # 데이터와 재구성된 데이터 크기가 일치하도록 변형
        reconstructions_resized = F.interpolate(reconstructions, size=data.shape[2:])

        # 재구성 오차 계산 (예: MSE)
        #error = torch.nn.functional.mse_loss(reconstructions, data)
        #errors.append(error.item())
        # 재구성 오차 계산 (예: MSE)
        #data_resized = data.view(reconstructions.shape)
        #error = torch.nn.functional.mse_loss(reconstructions, data_resized)
        #errors.append(error.item())

        # 재구성 오차 계산 (예: MSE)
        error = torch.nn.functional.mse_loss(reconstructions_resized, data)
        errors.append(error.item())

    return errors

# Deep SVDD 이상 탐지
def anomaly_detection(errors, threshold):
    return errors > threshold

def main():
    yolo_model_path = '/home/hkj/yolov10pj/yolov10_human/runs/detect/train4/weights/best.pt'
    svdd_model_path = '/home/hkj/yolov10pj/yolov10_human/pytorch-Deep-SVDD/weights/final_deep_svdd.pth'
    #image_path = '/home/hkj/yolov10pj/yolov10_human/dataset/test2/*.jpg'
    image_path = '/home/hkj/yolov10pj/yolov10_human/dataset/test2/printingtshirt5.jpg'

    # YOLO 모델 로딩 및 예측
    yolo_model = load_yolo_model(yolo_model_path)
    yolo_results = predict_yolo(yolo_model, image_path)
    
    # YOLO 예측 결과를 사용하여 Deep SVDD 모델에 입력할 데이터를 준비
    # 예를 들어, bounding box 좌표나 class label 등을 사용할 수 있습니다.
    # 여기서는 bounding box의 중심 좌표를 예로 사용합니다.
    #boxes = yolo_results[['xcenter', 'ycenter']].values
    #print(yolo_results)
    #for result in yolo_results:
    #    boxes = result.boxes
    boxes = yolo_results[0].boxes
    print(boxes)
    if len(boxes) == 0:
        print("No objects detected.")
        return

    # Deep SVDD 모델 로딩
    svdd_model = load_svdd_model(svdd_model_path)

    # 재구성 오차 계산
    #errors = compute_reconstruction_error(boxes, svdd_model)
    errors = compute_reconstruction_error(boxes, svdd_model, image_path)

    # 이상 탐지 임계값 설정
    threshold = 0.5  # 이 값은 모델의 특성에 맞게 조정해야 합니다.
    
    # 이상 탐지 결과
    anomalies = anomaly_detection(errors, threshold)
    print(f"Anomalies detected: {np.sum(anomalies)}")
    print(f"Errors: {errors}")
    print(f"Anomalies: {anomalies}")

if __name__ == "__main__":
    main()

