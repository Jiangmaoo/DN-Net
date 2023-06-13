import os

import cv2
import numpy as np


rootpath='./dataset/train/train_A'
outpath='./dataset/train/train_B/'

# def check_dir():
#     if not os.path.exists("../dataset/train/train_A"):
#         os.mkdir("./dataset/train/train_A")
#     if not os.path.exists("./dataset/train/train_B"):
#         os.mkdir("./dataset/train/train_B")
def processImage(filesource, destsource, name, imgtype):
    '''
        filesource是存放待雾化图片的目录
        destsource是存放物化后图片的目录
        name是文件名
        imgtype是文件类型
        '''
    imgtype = 'jpeg' if imgtype == '.jpg' else 'png'
    img=cv2.imread(name)

    # 设置雾的密度和透射率
    fog_density = 0.8
    transmission = 0.5

    # 计算雾的值
    height, width, _ = img.shape
    fog = np.zeros((height, width), dtype=np.float32)
    fog[:, :] = fog_density

    # 计算透射率
    trans_map = np.exp(-transmission * fog)

    # 添加雾
    fog_img = np.zeros_like(img, dtype=np.float32)
    for i in range(3):
        fog_img[:, :, i] = img[:, :, i] * trans_map + fog * (1 - trans_map)

    # 将图像转换回uint8格式
    fog_img = np.uint8(fog_img)
    cv2.imshow('fog_img',fog_img)
    cv2.imwrite(destsource+name,fog_img)
def run():
    #切换到源目录，遍历目录下所有图片
    os.chdir(rootpath)
    for i in os.listdir(os.getcwd()):
        #检查后缀
        postfix = os.path.splitext(i)[1]
        print(postfix,i)
        if postfix == '.jpg' or postfix == '.png':
            processImage(rootpath, outpath, i, postfix)

if __name__ == '__main__':
    # check_dir()
    run()