# 이미지 한장 테스트

from ultralytics import YOLOv10

#best_saved_model_path = '/home/hkj/yolov10pj/yolov10/trainresult/result_facedetect3/weights/best.pt' # s
#best_saved_model_path = '/home/hkj/yolov10pj/yolov10/trainresult/result_facedetect6/weights/best.pt' # n
#best_saved_model_path = '/home/hkj/yolov10pj/yolov10/trainresult/result_facedetect9/weights/best.pt' # m
#best_saved_model_path = '/home/hkj/yolov10pj/yolov10/trainresult/result_facedetect10/weights/best.pt' # b
#best_saved_model_path = '/home/hkj/yolov10pj/yolov10/trainresult/result_facedetect11/weights/best.pt' # l

#best_saved_model_path = '/home/hkj/yolov10pj/yolov10/trainresult/result_facedetect14/weights/best.pt' # x
#best_saved_model_path = '/home/hkj/yolov10pj/yolov10_human/trainresult/result_facedetect4/weights/best.pt' 
best_saved_model_path = '/home/hkj/yolov10pj/yolov10_human/trainresult/result_facedetect7/weights/best.pt'

# 모델 로드
model = YOLOv10(best_saved_model_path)

# 예측
#test_images_path = '/home/hkj/yolov10pj/yolov10_human/dataset/*.jpg'  
#test_images_path = '/home/hkj/yolov10pj/yolov10_human/dataset/test2/*.jpg'  
#test_images_path = '/home/hkj/yolov10pj/yolov10_human/dataset/test2/*.jpg'  
test_images_path = '/home/hkj/yolov10pj/yolov10_human/dataset/glass.jpg'  

model.predict(test_images_path, save=True, imgsz=640, conf=0.5, device=0)
#model.predict('/home/hkj/yolov10pj/yolov10_human/dataset/imoji_person.jpg', save=True, imgsz=640, conf=0.5, device=0)
#model.predict("/home/hkj/yolov10pj/yolov10/dataset/WIDER_DARKNET/images/test/4_Dancing_Dancing_4_876.jpg", save=True, imgsz=640, conf=0.5, device=0)
#model.predict("/home/hkj/yolov10pj/yolov10/dataset/WIDER_DARKNET/images/test/0_Parade_Parade_0_107.jpg", save=True, imgsz=640, conf=0.5, device=0)
#model.predict("/home/hkj/yolov10pj/yolov10/dataset/WIDER_DARKNET/images/test/3_Riot_Riot_3_434.jpg", save=True, imgsz=640, conf=0.5, device=0)
