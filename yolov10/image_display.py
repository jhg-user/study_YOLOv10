from PIL import Image

# OS 모듈을 import
import os


def display_jpeg_image(image_path):
    try:
        image = Image.open(image_path)
        image.show()
    except Exception as e:
        print("Error:", e)


root_dir = '/home/hkj/yolov10pj/yolov10/runs/detect/predict'
# os.listdir( ) 함수에 특정 디렉토리 경로 입력하여, 디렉토리 안의 파일들을 리스트로 저장.
file_list = os.listdir(root_dir)   # ex) os.listdir('/home/hello/test1')
print(file_list)

for file in file_list:
    jpeg_image_path = os.path.join(root_dir,file)  # 이미지 파일의 경로를 적절히 수정하세요
    print(jpeg_image_path)
    display_jpeg_image(jpeg_image_path)
