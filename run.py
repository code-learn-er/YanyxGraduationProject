import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from model import *
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from dataset import get_data
import torch.optim as optim
from torch import nn
from train import train_test
import os
import csv
from dataset import MyDataset
from torch.utils.data import DataLoader

# from PyQt5.QtCore import QLibraryInfo

# os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
#     QLibraryInfo.PluginsPath
# )
size = (256, 256)
sx_model_index = [3, 4, 3, 3]
mx_model_index = [3, 4, 3, 3]
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", default=["MultResNet", "MultCNN"], type=str
    )
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--lr", default=0.0005, type=float)
    parser.add_argument("--size", default=size, type=tuple)
    opt = parser.parse_args()
    model_names = opt.models
    batch_size = opt.batch_size
    epochs = opt.epochs
    lr = opt.lr
    size = opt.size

    root_dir = os.getcwd()
    weight_dir = os.path.join(root_dir, "weight")
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    if not os.path.exists("results"):
        os.makedirs("results")
    # 准备数据
    data = get_data(os.path.join(root_dir, "data"))
    # 数据转换器
    data_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    full_dataset = MyDataset(x=data, transform=data_transform)
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [int(0.9*len(full_dataset)), len(full_dataset) - int(0.9*len(full_dataset))])
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    # 可视化writer
    writer = SummaryWriter("logs")
    # 定义模型
    for model_name in model_names:
        xuxu = model[model_name](sx_model_index, mx_model_index).to(device)
        # if weight_dir is not None and model_name + ".pth" in os.listdir(weight_dir):
        #     try:
        #         xuxu.load_state_dict(
        #             torch.load(os.path.join(weight_dir, model_name + ".pth"))
        #         )
        #     except:
        #         print(
        #             "****(= 7 ^ 7 =)---- {}: model structure has been changed!!!".format(
        #                 model_name
        #             )
        #         )
        #     else:
        #         print(
        #             "\/\/\/(= ^ _ ^ =)//// {}: model load successfully !!!".format(
        #                 model_name
        #             )
        #         )

        # 定义优化器和误差函数
        optimizer = optim.Adam(xuxu.parameters(), lr=lr)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=8, last_epoch=-1)
        # lr_scheduler = optim.lr_scheduler.StepLR(
        #     optimizer, step_size=1, gamma=0.9)
        criterion = nn.CrossEntropyLoss().to(device)
        # 记录列表清零
        record=[[] for i in range(6)]

        for i in range(epochs):
            train_test(
                i,
                train_loader,
                test_loader,
                batch_size,
                data_transform,
                xuxu,
                optimizer,
                lr_scheduler,
                criterion,
                device,
                writer,
                record
            )

        torch.save(xuxu.state_dict(), os.path.join(
            weight_dir, model_name + ".pth"))

        
        train_acculist, train_F1list, train_losslist,test_acculist, test_F1list, test_losslist = record
        def plot(l, xlabel, ylabel, title):
            plt.plot(l)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title("{0}_bs{1}_Is{2}_{3}".format(
                model_name, batch_size, size[0], title))
            plt.savefig(
                "results/png/{0}_bs{1}_Is{2}_{3}.png".format(model_name, batch_size, size[0], title))
            plt.cla()
        plot(train_acculist, "epochs", "acc", "train_acc")
        plot(train_F1list, "epochs", "F1", "train_F1")
        plot(train_losslist, "epochs", "loss", "train_loss")
        plot(test_acculist, "epochs", "acc", "test_acc")
        plot(test_F1list, "epochs", "F1", "test_F1")
        plot(test_losslist, "epochs", "loss", "test_loss")

        def write(l, name):
            with open(
                "results/csv/{0}_bs{1}_Is{2}_{3}.csv".format(model_name, batch_size, size[0], name), "w", encoding="utf-8"
            ) as f:
                writercsv = csv.writer(f)
                writercsv.writerows([list(map(lambda x:round(x, 2), l))])
        write(train_acculist, "train_acc")
        write(train_F1list, "train_F1")
        write(train_losslist, "train_loss")
        write(test_acculist, "test_acc")
        write(test_F1list, "test_F1")
        write(test_losslist, "test_loss")

    writer.close()
