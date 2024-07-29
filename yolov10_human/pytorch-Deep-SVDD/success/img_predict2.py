import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLOv10
from model import network
import os
import glob

device = 'cuda:2'

# YOLO 모델 로딩
def load_yolo_model(model_path):
    with torch.no_grad():
        model = YOLOv10(model_path)
    return model

# Deep SVDD 모델 로딩
def load_svdd_model(path):
    net = network(z_dim=32).to(device)
    checkpoint = torch.load(path)
    net_dict = checkpoint['net_dict']
    net.load_state_dict(net_dict)
    net.eval()
    c = torch.tensor(checkpoint['center']).to(device)
    return net, c

# YOLO 모델을 사용하여 이미지에서 객체를 탐지
def predict_yolo(model, image_paths, batch_size):
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        try:
            batch_results = model.predict(batch_paths, save=False, imgsz=640, conf=0.5, device=device)
            results.extend(batch_results)
        except RuntimeError as e:
            print(f"YOLO 예측 중 런타임 오류 발생: {e}")
            print("이 배치에 대해 CPU로 전환합니다.")
            batch_results = model.predict(batch_paths, save=False, imgsz=640, conf=0.5, device='cpu')
            results.extend(batch_results)
        
        torch.cuda.empty_cache()
    return results

# 재구성 오차 계산 함수
def compute_reconstruction_error(boxes, model, image_path, c):
    errors = []
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # 모델 입력 크기에 맞게 조정
        transforms.ToTensor(),
        transforms.Grayscale(),  # 모델이 그레이스케일 입력을 기대하는 경우
    ])

    for box in boxes:
        xyxy = box.xyxy[0].tolist()
        left, top, right, bottom = map(int, xyxy[:4])

        # 이미지를 crop하고 Tensor로 변환
        cropped_image = image.crop((left, top, right, bottom))
        data = transform(cropped_image).unsqueeze(0).to(device)  # 배치 차원 추가 및 GPU로 이동

        # 모델을 사용하여 특성 추출 및 재구성 오차 계산
        z = model(data)
        error = torch.mean(torch.sum((z - c) ** 2, dim=1))
        errors.append(error.item())

    return errors

# 이상 탐지 결과 계산 함수
def anomaly_detection(errors, threshold):
    return ["Anomalous" if error > threshold else "Normal" for error in errors]

# 결과를 이미지에 표시
def draw_results(image_path, boxes, anomalies, output_path):
    try:
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)

        # 이미지 크기에 비례하는 글꼴 크기 계산
        image_width, image_height = image.size
        font_size = max(15, int(min(image_width, image_height) * 0.05))
        font = ImageFont.load_default()

        for i, box in enumerate(boxes):
            xyxy = box.xyxy[0].tolist()
            left, top, right, bottom = map(int, xyxy[:4])
            label = anomalies[i]

            # Bounding box 색상 설정
            color = 'blue' if label == 'Anomalous' else 'red'

            # Bounding box 그리기
            draw.rectangle([left, top, right, bottom], outline=color, width=3)
            draw.text((left, top), label, fill=color, font=font)

        path = image_path.split('/')[-1]
        output_image_path = os.path.join(output_path, path)

        # 파일 저장
        image.save(output_image_path)
        print(f'결과가 {output_image_path}에 저장되었습니다.')

    except Exception as e:
        print(f"파일 저장 중 오류 발생: {e}")
        print(f"{image_path}을(를) 처리할 수 없습니다. 이 파일을 건너뜁니다.")
'''
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)

    # 이미지 크기에 비례하는 글꼴 크기 계산
    image_width, image_height = image.size
    font_size = max(15, int(min(image_width, image_height) * 0.05))
    font = ImageFont.load_default()

    for i, box in enumerate(boxes):
        xyxy = box.xyxy[0].tolist()
        left, top, right, bottom = map(int, xyxy[:4])
        label = anomalies[i]

        # Bounding box 색상 설정
        color = 'blue' if label == 'Anomalous' else 'red'

        # Bounding box 그리기
        draw.rectangle([left, top, right, bottom], outline=color, width=3)
        draw.text((left, top), label, fill=color, font=font)

    path = image_path.split('/')[-1]
    output_image_path = os.path.join(output_path, path)

    image.save(output_image_path)
    print(f'결과가 {output_image_path}에 저장되었습니다.')
'''

def main():
    yolo_model_path = '/home/hkj/yolov10pj/yolov10_human/runs/detect/train4/weights/best.pt'
    svdd_model_path = '/home/hkj/yolov10pj/yolov10_human/pytorch-Deep-SVDD/weights/final_deep_svdd8.pth'
    image_folder = '/home/hkj/yolov10pj/yolov10_human/dataset/test/'
    output_folder = '/home/hkj/yolov10pj/yolov10_human/pytorch-Deep-SVDD/runs/predict_deep8'
    os.makedirs(output_folder, exist_ok=True)

    # 이미지 파일 목록 가져오기
    image_paths = glob.glob(os.path.join(image_folder, '*.jpg'))

    # YOLO 모델 로딩
    yolo_model = load_yolo_model(yolo_model_path)

    # Deep SVDD 모델 로딩
    svdd_model, c = load_svdd_model(svdd_model_path)

    batch_size = 2  # YOLO 모델 배치 크기
    yolo_results = predict_yolo(yolo_model, image_paths, batch_size)

    for i, image_path in enumerate(image_paths):
        boxes = yolo_results[i].boxes
        if len(boxes) == 0:
            torch.cuda.empty_cache()
            print(f"{image_path}에서 객체를 감지하지 못했습니다.")
            continue

        # 재구성 오차 계산
        errors = compute_reconstruction_error(boxes, svdd_model, image_path, c)

        # 이상 탐지 임계값 설정
        #threshold = 0.019278896506875753  # 기존의 0.5에서 더 낮춰 설정
        threshold = 0.003  # 기존의 0.5에서 더 낮춰 설정

        # 이상 탐지 결과
        anomalies = anomaly_detection(errors, threshold)
        print(f"{image_path}의 이상 탐지 결과: {anomalies}")
        print(errors)

        # 결과를 이미지에 표시
        draw_results(image_path, boxes, anomalies, output_folder)

        # GPU 메모리 비우기
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

