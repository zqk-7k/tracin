import os

import torch
import torch.nn as nn
import math
import time
from model import resnet34
from torchvision import transforms, datasets
from pif.influence_functions_new import get_gradient,tracin_get
from torch.autograd import grad
from data_get import dataset_get,dataset_category_get
import torchvision
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = {
    "train": transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

train_set = torchvision.datasets.CIFAR10(root='datasets', train=False,
                                         download=False, transform=data_transform["train"])
# 加载训练集，实际过程需要分批次（batch）训练
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1,
                                           shuffle=False, num_workers=0)

test_set = torchvision.datasets.CIFAR10(root='datasets', train=False,
                                            download=False, transform=data_transform["val"])
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                          shuffle=False, num_workers=0)
for classnum in range(10):
    test_data_iter = iter(test_loader)
    test_image, test_label = test_data_iter.next()
    while (1):
        if (test_label[0] == classnum):
            break
        else:
            test_image, test_label = test_data_iter.next()

    # #提取test image
    # test_extract = torch.zeros(1,3,224,224)
    # test_extract[0] = test_image[0]
    # test_extract[0][0][0][0] = 10

    net = resnet34()
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 10)
    net.to(device)

    #############################################
    # 提取train前1000个label
    train_set2 = torchvision.datasets.CIFAR10(root='datasets', train=False,
                                             download=False, transform=data_transform["train"])
    # 加载训练集，实际过程需要分批次（batch）训练
    train_loader2 = torch.utils.data.DataLoader(train_set2, batch_size = 1000,
                                               shuffle=False, num_workers=0)
    train_data_iter1000 = iter(train_loader2)
    train_image1000, train_label1000 = train_data_iter1000.next()
    print(train_label1000[10])

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    train_bar = tqdm(train_loader)
    rank = [[], [], [], [], []]

    score_list = []
    for i in range(4):
        print(i)

        for j in [60, 70, 80, 90, 100]:
            model_weight_path = './pth_file4_1000/resNet34_tracin_epoch{}_{}.pth'
            net.load_state_dict(torch.load(model_weight_path.format(i, j), map_location=device))
            net.to(device)

            logits_test = net(test_image.to(device))
            loss_test = loss_function(logits_test, test_label.to(device))
            grad_z_test = grad(loss_test, net.parameters())
            grad_z_test = get_gradient(grad_z_test, net)

            time_start = time.perf_counter()

            for step, data in enumerate(train_bar):
                if (step < 1000):
                    images, labels = data

                    if(labels[0] == classnum):
                        logits_train = net(images.to(device))
                        loss_train = loss_function(logits_train, labels.to(device))
                        grad_z_train = grad(loss_train, net.parameters())
                        grad_z_train = get_gradient(grad_z_train, net)

                        score = tracin_get(grad_z_test, grad_z_train)
                        if (j == 60 and i == 0):
                            score_list.append(float(score))
                        else:
                            score_list[step] = score_list[step] + float(score)


                    else:
                        if (j == 60 and i == 0):
                            score_list.append(float(100000.0))
                        else:
                            score_list[step] = score_list[step]
                    if (j == 100 and step == 999):
                        # score_list存储tracin得分组
                        # print(score_list)
                        score_list_copy = []
                        for p in range(len(score_list)):
                            score_list_copy.append(score_list[p])
                        score_list_copy.sort()
                        for p in range(len(score_list)):
                            for q in range(len(score_list)):
                                if (score_list[p] == score_list_copy[q]):
                                    rank[i].append(q + 1)
                                    break
    print(rank[3])
    print('####################################')
    # top10 = []
    # amount10 = 0
    # top20 = []
    # amount20 = 0
    # top40 = []
    # amount40 = 0
    # top80 = []
    # amount80 = 0
    # for i in range(len(rank[3])):
    #     if (rank[3][i] < 990):
    #         top10.append(i)
    #         amount10 += 1
    #     if (rank[3][i] < 980):
    #         top20.append(i)
    #         amount20 += 1
    #     if (rank[3][i] < 960):
    #         top40.append(i)
    #         amount40 += 1
    #     if (rank[3][i] < 920):
    #         top80.append(i)
    #         amount80 += 1
    # print(top10)
    # print(amount10)
    # print(top20)
    # print(amount20)
    # print(top40)
    # print(amount40)
    # print(top80)
    # print(amount80)
    # labels_record = []
    # for i in range(1000):
    #     if (rank[3][i] > 900):
    #         labels_record.append(train_label1000[i])
    # number = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # for i in range(len(labels_record)):
    #     number[labels_record[i]] = number[labels_record[i]] + 1
    #
    # print('####################################')
    # print(number)

#
#                     # print(min(score_list),max(score_list))
# rank_difference = []
# for i in range(1,4):
#     difference = 0.0
#     difference2 = 0.0
#
#     for j in range(100):
#         difference = difference + math.fabs(rank[i][j] - rank[0][j])
#
#         difference2 = difference2 + math.fabs(rank[i][j] - rank[i-1][j])
#     difference = difference/100
#     print(difference2/100)
#     rank_difference.append(difference)
# print(rank_difference)
# #rank_difference是epoch2，3，4，5与epoch1的区别，difference2是
#
# print('%f s' % (time.perf_counter() - time_start))
# # print(score_list)