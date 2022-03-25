import os

import torch
import torch.nn as nn

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
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

train_set = torchvision.datasets.CIFAR10(root='datasets', train=False,
                                         download=False, transform=data_transform["train"])
# 加载训练集，实际过程需要分批次（batch）训练
train_loader = torch.utils.data.DataLoader(train_set, batch_size=5,
                                           shuffle=False, num_workers=0)

test_set = torchvision.datasets.CIFAR10(root='datasets', train=False,
                                            download=False, transform=data_transform["val"])
test_loader = torch.utils.data.DataLoader(test_set, batch_size=50,
                                          shuffle=False, num_workers=0)
test_data_iter = iter(test_loader)
test_image, test_label = test_data_iter.next()

net = resnet34()
in_channel = net.fc.in_features
net.fc = nn.Linear(in_channel, 10)
net.to(device)



# define loss function
loss_function = nn.CrossEntropyLoss()

train_bar = tqdm(train_loader)
score_list = []
for j in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    model_weight_path = "pth_file5_500/resNet34_tracin_{}.pth"
    net.load_state_dict(torch.load(model_weight_path.format(j), map_location=device))
    net.to(device)

    logits_test = net(test_image.to(device))
    loss_test = loss_function(logits_test, test_label.to(device))
    grad_z_test = grad(loss_test, net.parameters())
    grad_z_test = get_gradient(grad_z_test, net)

    time_start = time.perf_counter()

    for step, data in enumerate(train_bar):
        if (step < 100):
            images, labels = data

            logits_train = net(images.to(device))
            loss_train = loss_function(logits_train, labels.to(device))
            grad_z_train = grad(loss_train, net.parameters())
            grad_z_train = get_gradient(grad_z_train, net)

            score = tracin_get(grad_z_test, grad_z_train)
            if (j == 10):
                score_list.append(float(score))
            else:
                score_list[step] = score_list[step] + float(score)

            if (j == 100 and step == 99):
                # score_list存储tracin得分组
                # print(score_list)
                score_list_copy = []
                for p in range(len(score_list)):
                    score_list_copy.append(score_list[p])
                score_list_copy.sort()
                rank = [] 
                for p in range(len(score_list)):
                    for q in range(len(score_list)):
                        if (score_list[p] == score_list_copy[q]):
                            rank.append(q + 1)
                            break
                # rank储存tracin从小到大的位次
                print(rank)

                # print(min(score_list),max(score_list))


print('%f s' % (time.perf_counter() - time_start))
# print(score_list)