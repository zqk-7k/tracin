import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm


def dataset_get():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     #transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}



    # 10000张测试图片
    test_set = torchvision.datasets.CIFAR10(root='datasets', train=False,
                                            download=False, transform=data_transform["val"])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100,
                                              shuffle=False, num_workers=0)
    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 数据处理
    transform2 = transforms.Compose(
        [transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0, hue=0),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = torchvision.datasets.CIFAR10(root='datasets', train=True,
                                             download=False, transform=data_transform["train"])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=500,
                                               shuffle=False, num_workers=0)
    train_data_iter = iter(train_loader)
    train_image, train_label = train_data_iter.next()

    img_all_train = torch.zeros(10, 50, 3, 224, 224)
    label_all_train = torch.zeros(10, 50)
    cls_num_train = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(50):
        img_all_train[train_label[i]][cls_num_train[train_label[i]]] = train_image[i]
        label_all_train[train_label[i]][cls_num_train[train_label[i]]] = train_label[i]
        cls_num_train[train_label[i]] += 1

    test_data_iter = iter(test_loader)
    test_image, test_label = test_data_iter.next()

    img_all_test = torch.zeros(10, 100, 3, 224, 224)
    label_all_test = torch.zeros(10, 100)
    cls_num_test = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(10):
        img_all_test[test_label[i]][cls_num_test[test_label[i]]] = test_image[i]
        label_all_test[test_label[i]][cls_num_test[test_label[i]]] = test_label[i]
        cls_num_test[test_label[i]] += 1
    return img_all_train,label_all_train,img_all_test,label_all_test #返回shape(10, 50, 3, 224, 224),(10, 50),(10, 100, 3, 224, 224),(10, 100)


def dataset_category_get(category_num):
    data_transform = {
        "train": transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}



    # 10000张测试图片
    test_set = torchvision.datasets.CIFAR10(root='datasets', train=False,
                                            download=False, transform=data_transform["val"])
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=2000,
                                              shuffle=False, num_workers=0)

    train_set = torchvision.datasets.CIFAR10(root='datasets', train=False,
                                             download=False, transform=data_transform["train"])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=10000,
                                               shuffle=False, num_workers=0)
    train_data_iter = iter(train_loader)
    train_image, train_label = train_data_iter.next()

    img_all_train = torch.zeros(500, 3, 224, 224)
    train_image_num = 0
    for i in range(10000):
        if(train_label[i] == category_num):
            img_all_train[train_image_num] = train_image[i]
            train_image_num += 1
        if(train_image_num == 500 ):
            break

    test_data_iter = iter(test_loader)
    test_image, test_label = test_data_iter.next()

    img_all_test = torch.zeros(100, 3, 224, 224)
    test_image_num = 0
    for i in range(2000):
        if (test_label[i] == category_num):
            img_all_test[test_image_num] = test_image[i]
            test_image_num += 1
        if (test_image_num == 100):
            break

    return img_all_train,img_all_test #返回shape(500, 3, 224, 224),(100, 3, 224, 224)