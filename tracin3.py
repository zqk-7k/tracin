import os

import torch
import torch.nn as nn

import time
from model import resnet34

from pif.influence_functions_new import get_gradient,tracin_get
from torch.autograd import grad
from data_get import dataset_get,dataset_category_get

category_num = 0
img_all_train,img_all_test = dataset_category_get(0)
img_all_train = img_all_train.view(500, 1, 3, 224, 224)
img_all_test = img_all_test.view(100, 1, 3, 224, 224)


img_all_train2,img_all_test2 = dataset_category_get(4)
img_all_train2 = img_all_train2.view(500, 1, 3, 224, 224)
img_all_test2 = img_all_test2.view(100, 1, 3, 224, 224)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = resnet34()
in_channel = net.fc.in_features
net.fc = nn.Linear(in_channel, 10)
net.to(device)
# load pretrain weights
# download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
# model_weight_path = "pth_file/resNet34_epoch1.pth"
# model_weight_path = "pth_file2/resNet34_tracin_120.pth"
# assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
# net.load_state_dict(torch.load(model_weight_path, map_location=device))
# # for param in net.parameters():
# #     param.requires_grad = False
#
# net.to(device)


# define loss function
loss_function = nn.CrossEntropyLoss()


for k in range(10):
    score_list = []
    for j in [10,20,30,40,50,60,70,80,90,100]:
        model_weight_path = "pth_file1/resNet34_tracin_{}.pth"
        net.load_state_dict(torch.load(model_weight_path.format(j), map_location=device))
        net.to(device)

        label_test = torch.zeros(1).long()
        label_test[0] = 4
        logits_test = net(img_all_test2[k].to(device))
        loss_test = loss_function(logits_test, label_test.to(device))
        grad_z_test = grad(loss_test, net.parameters())
        grad_z_test = get_gradient(grad_z_test, net)


        time_start = time.perf_counter()
        for i in range(50):
            label_train = torch.zeros(1).long()
            label_test[0] = 0
            logits_train = net(img_all_train[i].to(device))
            loss_train = loss_function(logits_train, label_train.to(device))
            grad_z_train = grad(loss_train, net.parameters())
            grad_z_train = get_gradient(grad_z_train, net)

            score = tracin_get(grad_z_test, grad_z_train)
            if(j == 10):
                score_list.append(float(score))
            else:
                score_list[i] = score_list[i] + float(score)
            if(j == 100 and i ==49):
                #score_list存储tracin得分组
                # print(score_list)
                score_list_copy = []
                for p in range(len(score_list)):
                    score_list_copy.append(score_list[p])
                score_list_copy.sort()
                rank = []
                for p in range(len(score_list)):
                    for q in range(len(score_list)):
                        if(score_list[p] == score_list_copy[q]):
                            rank.append(q+1)
                            break
                #rank储存tracin从小到大的位次
                print(rank)

                # print(min(score_list),max(score_list))



print('%f s' % (time.perf_counter() - time_start))
# print(score_list)