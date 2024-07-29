# 이미지 한장 테스트

from ultralytics import YOLOv10

#best_saved_model_path = '/home/hkj/yolov10pj/yolov10/trainresult/result_facedetect3/weights/best.pt' # s
#best_saved_model_path = '/home/hkj/yolov10pj/yolov10/trainresult/result_facedetect6/weights/best.pt' # n
#best_saved_model_path = '/home/hkj/yolov10pj/yolov10/trainresult/result_facedetect9/weights/best.pt' # m
#best_saved_model_path = '/home/hkj/yolov10pj/yolov10/trainresult/result_facedetect10/weights/best.pt' # b
best_saved_model_path = '/home/hkj/yolov10pj/yolov10/trainresult/result_facedetect13/weights/best.pt' # l
#best_saved_model_path = '/home/hkj/yolov10pj/yolov10/trainresult/result_facedetect14/weights/best.pt' # x

# 모델 로드
model = YOLOv10(best_saved_model_path)

# 예측
#model.predict("/home/hkj/yolov10pj/yolov10/dataset/doll.jpg", save=True, imgsz=640, conf=0.5, device=0)
#model.predict("/home/hkj/yolov10pj/yolov10/dataset/*.jpg", save=True, imgsz=640, conf=0.5, device=0)
#model.predict("/home/hkj/yolov10pj/yolov10/dataset/*.jpg", save=True, conf=0.5, device=0)
#model.predict("/home/hkj/yolov10pj/yolov10/dataset/imoji2.jpg", save=True, conf=0.5, imgsz=640, device=0)

#model.predict("/home/hkj/yolov10pj/yolov10/dataset/imoji_person.jpg", save=True, conf=0.5, imgsz=640, device=0)
#model.predict("/home/hkj/yolov10pj/yolov10/dataset/neoguri.jpg", save=True, conf=0.5, imgsz=640, device=0)
#model.predict("/home/hkj/yolov10pj/yolov10/dataset/rabbit.jpg", save=True, conf=0.5, imgsz=640, device=0)

#model.predict("/home/hkj/yolov10pj/yolov10/dataset/test3/*.jpg", imgsz=640, save=True, conf=0.5, device=0)
#model.predict("/home/hkj/yolov10pj/yolov10/dataset/test3/printingtshirt7.jpg", imgsz=640, save=True, conf=0.5, device=0)
#model.predict("/home/hkj/yolov10pj/yolov10/dataset/test3/printingtshirt8.jpg", imgsz=640, save=True, conf=0.5, device=0)
results = model.predict("/home/hkj/yolov10pj/yolov10/dataset/bigface/bigface.jpg", save=True, conf=0.5, device=0)
#model.predict("/home/hkj/yolov10pj/yolov10/dataset/test3/jjangu.jpg", imgsz=640, save=True, conf=0.5, device=0)
#model.predict("/home/hkj/yolov10pj/yolov10/dataset/test3/mario.jpg", imgsz=640, save=True, conf=0.5, device=0)
#model.predict("/home/hkj/yolov10pj/yolov10/dataset/test3/ppukka.jpg", imgsz=640, save=True, conf=0.5, device=0)
#model.predict("/home/hkj/yolov10pj/yolov10/dataset/test3/webtoon.jpg", imgsz=640, save=True, conf=0.5, device=0)
#model.predict("/home/hkj/yolov10pj/yolov10/dataset/test3/webtoonandperson.jpg", imgsz=640, save=True, conf=0.5, device=0)

# 결과에서 바운딩 박스 좌표 가져오기
boxes = results[0].boxes

# 바운딩 박스 좌표를 정규화하여 .txt 파일로 저장
with open('dataset/bigface/origin.txt', 'w') as f:
    for box in boxes:
        x1, y1, w, h = box.xywhn[0].tolist()  # 좌표를 리스트로 변환하여 가져오기
        class_id = box.cls[0].item()  # 텐서를 Python 숫자로 변환

        class_id = int(class_id)

        # Write to txt file in YOLO format
        f.write(f"{class_id} {x1} {y1} {w} {h}\n")

