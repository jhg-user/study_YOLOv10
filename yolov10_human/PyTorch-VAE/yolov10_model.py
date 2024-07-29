from ultralytics import YOLOv10
import cv2

def load_yolov10_model(yolov10_checkpoint_path):
    model = YOLOv10(yolov10_checkpoint_path)
    return model

def detect_faces_yolo(model, img):
    results = model.predict(img, imgsz=640, conf=0.5, device=0)
    return results

