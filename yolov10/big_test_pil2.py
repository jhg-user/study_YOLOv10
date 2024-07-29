# 라이브러리 import 
import cv2
import numpy as np
import os
import glob

# 이미지 파일 목록을 취득(패스)
pics=glob.glob('/home/hkj/yolov10pj/yolov10/dataset/bigface/*.jpg')

# 조정 후 사이즈를 지정(베이스 이미지)
size=(256,256)

# 리사이즈 처리
for pic in pics:
    base_pic=np.zeros((size[1],size[0],3),np.uint8)
    pic1=cv2.imread(pic,cv2.IMREAD_COLOR)
    h,w=pic1.shape[:2]
    ash=size[1]/h
    asw=size[0]/w
    if asw<ash:
        sizeas=(int(w*asw),int(h*asw))
    else:
        sizeas=(int(w*ash),int(h*ash))
    pic1 = cv2.resize(pic1,dsize=sizeas)
    base_pic[int(size[1]/2-sizeas[1]/2):int(size[1]/2+sizeas[1]/2),
    int(size[0]/2-sizeas[0]/2):int(size[0]/2+sizeas[0]/2),:]=pic1
    cv2.imwrite(new_fol+'/'+pic,base_pic)
