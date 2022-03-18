import os

import torch
import torch.nn as nn
import math

from model import resnet34

from pif.influence_functions_new import get_gradient,tracin_get
from torch.autograd import grad
from data_get import dataset_get,dataset_category_get
from image_process import get_mean,image_mask_single,image_mask_all

category_num = 0
img_all_train,img_all_test = dataset_category_get(1)

for i in range(10):
    print(img_all_train[0][0][0][i])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = resnet34()
in_channel = net.fc.in_features
net.fc = nn.Linear(in_channel, 10)
net.to(device)

# load pretrain weights
# download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
model_weight_path = "pth_file/resNet34_epoch1.pth"
assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
net.load_state_dict(torch.load(model_weight_path, map_location=device))
# for param in net.parameters():
#     param.requires_grad = False

net.to(device)



# define loss function
loss_function = nn.CrossEntropyLoss()

img_all_train = img_all_train.view(500, 1, 3, 224, 224)
img_all_test = img_all_test.view(100, 1, 3, 224, 224)

label_train = torch.zeros(1).long()
logits_train = net(img_all_train[0].to(device))
loss_train = loss_function(logits_train, label_train.to(device))
grad_z_train = grad(loss_train, net.parameters())
grad_z_train = get_gradient(grad_z_train, net)


label_test = torch.zeros(1).long()
label_test[0] = 0
logits_test = net(img_all_test[0].to(device))
loss_test = loss_function(logits_test, label_test.to(device))
grad_z_test = grad(loss_test, net.parameters())
grad_z_test = get_gradient(grad_z_test, net)

score_tracin = tracin_get(grad_z_test, grad_z_train)
print(score_tracin)

Pos_num = 0 #score_mask-score_mask_unimp>0的数量
abs = 0
# score_list = []
for i in range(1):
    ###############################################
    # MASK重要部分
    pointx = 50
    pointy = 50
    w = h = 100
    mask = get_mean(img_all_test[i])
    # mask = [0,0,0]

    img_mask = torch.rand(1, 3, 224, 224)
    img_mask = img_all_test[i]
    img_mask = image_mask_single(img_mask, pointx, pointy, w, h, mask)

    label_mask = torch.zeros(1).long()
    label_mask[0] = 0
    img_mask = img_mask.view(-1, 3, 224, 224)
    logits_mask = net(img_mask.to(device))
    loss_mask = loss_function(logits_mask, label_mask.to(device))
    grad_z_mask = grad(loss_mask, net.parameters())
    grad_z_mask = get_gradient(grad_z_mask, net)

    score_mask = tracin_get(grad_z_mask, grad_z_train)
    # print(score_mask)

    ###############################################
    # MASK次要部分
    pointx = 0
    pointy = 0
    w = h = 75
    mask = get_mean(img_all_test[i])
    # mask = [0,0,0]

    img_mask = torch.rand(1, 3, 224, 224)
    img_mask = img_all_test[i]
    img_mask = image_mask_single(img_mask, pointx, pointy, w, h, mask)

    label_mask = torch.zeros(1).long()
    label_mask[0] = 0
    img_mask = img_mask.view(-1, 3, 224, 224)
    logits_mask = net(img_mask.to(device))
    loss_mask = loss_function(logits_mask, label_mask.to(device))
    grad_z_mask = grad(loss_mask, net.parameters())
    grad_z_mask = get_gradient(grad_z_mask, net)

    score_mask_unimp = tracin_get(grad_z_mask, grad_z_train)
    print(score_mask-score_mask_unimp)
    if(score_mask-score_mask_unimp > 0):
        Pos_num += 1
    abs += math.fabs(score_mask-score_mask_unimp)


print(Pos_num)
print(abs)