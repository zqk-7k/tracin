import numpy as np
import torch

def get_mean(img):
    #获取图像的均值
    img = img.view(-1,3,224,224)
    img_num = len(img)
    # mean = np.random.rand(img_num,3)
    for i in range(img_num):
        mean = torch.sum(img, 2)
        mean = torch.sum(mean, 2)
    mean = mean / (224*224)
    return mean #返回值(x,3)


def image_mask_single(img,pointx,pointy,w,h,mask):
    #单个图像的mask覆盖
    image = img.view(3,224,224)
    if(len(mask)==1):
        for i in range(w):
            for j in range(h):
                image[0][pointx + i][pointy + j] = mask[0][0]
                image[1][pointx + i][pointy + j] = mask[0][1]
                image[2][pointx + i][pointy + j] = mask[0][2]
    else:
        for i in range(w):
            for j in range(h):
                image[0][pointx + i][pointy + j] = mask[0]
                image[1][pointx + i][pointy + j] = mask[1]
                image[2][pointx + i][pointy + j] = mask[2]
    return image


def image_mask_all(img,pointx,pointy,w,h,mask):
    # 多个图像的mask覆盖
    img = img.view(-1,3,224,224)
    for k in range(len(img)):
        for i in range(w):
            for j in range(h):
                img[k][0][pointx + i][pointy + j] = mask[k][0]
                img[k][1][pointx + i][pointy + j] = mask[k][1]
                img[k][2][pointx + i][pointy + j] = mask[k][2]
    return img

