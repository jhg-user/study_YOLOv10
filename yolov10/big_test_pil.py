from PIL import Image
import os

root_dir = '/home/hkj/yolov10pj/yolov10/dataset/bigface'
imglist = os.listdir(root_dir) 

basewidth = 500

for idx, img in enumerate(imglist):
    try:
        imgor = Image.open(root_dir + img + '.jpg')
            wpercent = (baseheight/float(imgor.size[0])) 
            hsize = int((float(imgor.size[1])*float(hpercent))) 
            imgrs = imgor.resize((basewidth, hsize), Image.ANTIALIAS)
            imgrs.save(root_dir + img + '.jpg')  
    except Exception as e:
        print(e)
