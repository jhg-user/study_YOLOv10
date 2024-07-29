import os
import glob
import cv2

# 원본 이미지 경로
image_dir = '/home/hkj/yolov10pj/yolov10_human/pytorch-Deep-SVDD/human_images/*jpg'
image_paths = glob.glob(image_dir)

# 텍스트 파일 경로
txt_dir = '/home/hkj/yolov10pj/yolov10_human/runs/annot_humanbbox'

# 결과 저장 폴더 설정
output_dir = '/home/hkj/yolov10pj/yolov10_human/pytorch-Deep-SVDD/crop_human_images'
os.makedirs(output_dir, exist_ok=True)

# YOLO 형식 바운딩 박스 좌표를 이미지 크기와 함께 사용해 절대 좌표로 변환
def yolo_to_bbox(yolo_bbox, img_width, img_height):
    x_center, y_center, width, height = yolo_bbox
    x_center, y_center, width, height = float(x_center), float(y_center), float(width), float(height)

    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height

    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)

    return x_min, y_min, x_max, y_max

# 이미지와 대응되는 텍스트 파일 이름 설정
for image_path in image_paths:
    img_name = os.path.basename(image_path)
    txt_path = os.path.join(txt_dir, img_name.replace('.jpg', '.txt'))

    # 텍스트 파일 존재 여부 확인
    if not os.path.isfile(txt_path):
        continue

    # 텍스트 파일 읽기
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    # 원본 이미지 로드
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]

    # 클래스 ID가 0인 바운딩 박스 처리
    class_id_0_bboxes = []

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        bbox = parts[1:]

        if class_id == 0:
            x_min, y_min, x_max, y_max = yolo_to_bbox(bbox, img_width, img_height)
            class_id_0_bboxes.append((x_min, y_min, x_max, y_max))

    # 바운딩 박스가 있는 경우 크롭 및 저장
    if class_id_0_bboxes:
        for idx, (x_min, y_min, x_max, y_max) in enumerate(class_id_0_bboxes):
            cropped_img = img[y_min:y_max, x_min:x_max]

            # 파일명 결정
            if len(class_id_0_bboxes) == 1:
                cropped_img_path = os.path.join(output_dir, img_name)
            else:
                base_name, ext = os.path.splitext(img_name)
                cropped_img_path = os.path.join(output_dir, f"{base_name}_{idx + 1}{ext}")

            # 크롭한 이미지 저장
            cv2.imwrite(cropped_img_path, cropped_img)

print("Cropping and saving images completed successfully.")

