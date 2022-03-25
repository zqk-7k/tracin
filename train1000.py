import os
import json
import torchvision
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model import resnet34

category = [[0,1,2,8,9],[3,4,5,6,7]]
rank = [61, 37, 19, 55, 88, 71, 94, 68, 98, 95, 48, 43, 31, 32, 84, 41, 21, 42, 1, 67, 5, 96, 4, 3, 70, 56, 45, 53,
        91, 78, 89, 2, 79, 18, 90, 60, 15, 82, 11, 83, 13, 36, 28, 62, 51, 80, 72, 29, 65, 22, 44, 38, 46, 77, 35,
        66, 40, 12, 87, 97, 57, 9, 34, 49, 24, 93, 99, 74, 26, 81, 63, 27, 86, 7, 58, 14, 75, 30, 6, 100, 16, 39, 23,
        73, 10, 92, 76, 20, 54, 17, 52, 69, 50, 64, 25, 59, 33, 85, 8, 47]
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_set = torchvision.datasets.CIFAR10(root='datasets', train=True,
                                             download=False, transform=data_transform["train"])
    # 加载训练集，实际过程需要分批次（batch）训练
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=5,
                                               shuffle=False, num_workers=0)

    # 10000张测试图片
    test_set = torchvision.datasets.CIFAR10(root='datasets', train=False,
                                            download=False, transform=data_transform["val"])
    val_num = len(test_set)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=10,
                                              shuffle=False, num_workers=0)

    net = resnet34()
    # load pretrain weights
    # download url: https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weight_path = "pth_file1/resnet34-pre.pth"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location=device))
    # for param in net.parameters():
    #     param.requires_grad = False

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 10)
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs = 5
    best_acc = 0.0
    save_path = './pth_file4_1000/resNet34_tracin_epoch{}_{}.pth'  # save_path = './resNet34_new.pth'
    # save_path = './pth_file2/resNet34_epoch1.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        time_start = time.perf_counter()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            if (step <= 100):
                images, labels = data

                optimizer.zero_grad()
                logits = net(images.to(device))
                loss = loss_function(logits, labels.to(device))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()

            if (step <= 100 and step % 10 == 0 and step != 0):
                # torch.save(net.state_dict(), save_path + str(step) + '.pth')
                torch.save(net.state_dict(), save_path.format(epoch,step))
            # if (epoch == 0):
            #     if (step < 100 and rank[step] > i):
            #         images, labels = data
            #
            #         optimizer.zero_grad()
            #         logits = net(images.to(device))
            #         loss = loss_function(logits, labels.to(device))
            #         loss.backward()
            #         optimizer.step()
            #
            #         # print statistics
            #         running_loss += loss.item()
            # else:
            #     if (step < 100 and rank[step] > (100-i)):
            #         images, labels = data
            #
            #
            #         optimizer.zero_grad()
            #         logits = net(images.to(device))
            #         loss = loss_function(logits, labels.to(device))
            #         loss.backward()
            #         optimizer.step()
            #
            #         # print statistics
            #         running_loss += loss.item()

        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(test_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

            val_accurate = acc / 10000
            print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %  # 打印epoch，step，loss，accuracy
                  (epoch + 1, step + 1, running_loss / 500, val_accurate))

            print('%f s' % (time.perf_counter() - time_start))  # 打印耗时
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        if val_accurate > best_acc:
            best_acc = val_accurate
            # torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
