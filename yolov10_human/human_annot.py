import glob
import cv2
import os
import torch
from ultralytics import YOLOv10

# 모델 경로
#best_saved_model_path = '/home/hkj/yolov10pj/yolov10/trainresult/result_facedetect13/weights/best.pt' # class 1개 학습된 l
#best_saved_model_path = '/home/hkj/yolov10pj/yolov10_human/trainresult/result_facedetect4/weights/best.pt' # class 2개 학습된 s
best_saved_model_path = '/home/hkj/yolov10pj/yolov10_human/trainresult/result_facedetect7/weights/best.pt' # class 2개 학습된 l

# 모델 로드
model = YOLOv10(best_saved_model_path)

# 테스트 이미지 경로 설정
#test_images_path = '/home/hkj/yolov10pj/yolov10_human/dataset/pexels_faces/*.jpg'  # bigface 경로로
#test_images_path = '/home/hkj/yolov10pj/yolov10_human/dataset/unsplash_faces/*.jpg'  # bigface 경로로
#test_images_path = '/home/hkj/yolov10pj/yolov10_human/dataset/UTKface/*.jpg'  
#test_images_path = '/home/hkj/yolov10pj/yolov10_human/pytorch-Deep-SVDD/dataset/train/human-face/*jpg'  
test_images_path = '/home/hkj/yolov10pj/yolov10_human/pytorch-Deep-SVDD/human_images/*jpg'  

# 결과 저장 폴더 설정
output_dir = 'runs/annot_humanbbox'
#output_dir = 'nonhuman_list'
#output_dir = 'runs/predict_pexel'
#output_dir = 'runs/predict_UTK'
#output_dir = 'runs/predict_unsplash'
os.makedirs(output_dir, exist_ok=True)

# 배치 크기 설정
batch_size = 2  # 한 번에 처리할 이미지 수

# 모든 테스트 이미지 경로 가져오기
image_paths = glob.glob(test_images_path)

# 배치 단위로 예측 수행
for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i + batch_size]
    images = []
    original_dims = []


    # 배치 내 각 이미지 처리
    for img_path in batch_paths:
        # 이미지 불러오기
        image = cv2.imread(img_path)

        # 현재 이미지 크기 얻기
        original_height, original_width = image.shape[:2]

        images.append(image)
        original_dims.append((original_height, original_width))


    # 배치 내 각 이미지 처리
    #for img_path in batch_paths:
    #    images.append(img_path)

    # YOLOv10 모델 예측
    with torch.no_grad():
        try:
            #results = model.predict(images, save=True, imgsz=imgszs[0], conf=0.5, device=1)
            results = model.predict(images, save=False, imgsz=640, conf=0.5, device=1)
        except RuntimeError as e:
            print(f"RuntimeError occurred: {e}")
            print("Switching to CPU for this batch.")
            results = model.predict(images, save=False, imgsz=640, conf=0.5, device='cpu')

    # 메모리 클리어
    torch.cuda.empty_cache()

    # 결과 저장
    for img_path, result, (original_height, original_width) in zip(batch_paths, results, original_dims):
        boxes = result.boxes
        annot_file = img_path.split('/')[-1].replace('jpg', 'txt')
        annot_file = os.path.join(output_dir, annot_file)
        with open(annot_file, 'w') as f:
            for box in boxes:
                x, y, w, h = box.xywhn[0].tolist()
                score = box.conf[0].item()
                class_id = box.cls[0].item()

                x1 = (x - w / 2) * original_width
                y1 = (y - h / 2) * original_height
                x2 = (x + w / 2) * original_width
                y2 = (y + h / 2) * original_height

                x_center = (x1 + x2) / (2 * original_width)
                y_center = (y1 + y2) / (2 * original_height)
                width = (x2 - x1) / original_width
                height = (y2 - y1) / original_height
                class_id = int(class_id)

                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        print(f"Annotations saved for {img_path}")

print("Bounding boxes annotations saved successfully")


