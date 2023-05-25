from sklearn.model_selection import KFold
import torch
from torch.utils.tensorboard import SummaryWriter
from model import MultCNN
import random

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import MyDataset, get_data
import torch.optim as optim
from torch import nn
esp = 1e-6
def get_num_correct(preds, labels,TP_TN_FP_FN):
    preds = preds.argmax(dim=1)
    for i in range(len(preds)):
        if preds[i] == 0 and labels[i] == 0:
            TP_TN_FP_FN[0] += 1
        if preds[i] == 1 and labels[i] == 1:
            TP_TN_FP_FN[1] += 1
        if preds[i] == 0 and labels[i] == 1:
            TP_TN_FP_FN[2] += 1
        if preds[i] == 1 and labels[i] == 0:
            TP_TN_FP_FN[3] += 1
def get_acc_F1(TP_TN_FP_FN):
    TP,TN,FP,FN = TP_TN_FP_FN
    P = TP / (TP + FP + esp)
    R = TP / (TP + FN + esp)
    F1 = 2 * P * R / (P + R + esp)
    acc = (TP + TN) / (TP + TN + FP + FN + esp)
    return acc,F1
# 将data中取出train和test的dataset，再分别转化为dataloader
def get_train_test_dataloader(
    data, batch_size, train_index, test_index, data_transform
):
    train_fold = torch.utils.data.dataset.Subset(data, train_index)
    test_fold = torch.utils.data.dataset.Subset(data, test_index)

    train_fold = MyDataset(x=train_fold, transform=data_transform)
    test_fold = MyDataset(x=test_fold, transform=data_transform)

    # 打包成DataLoader类型用于训练
    train_loader = DataLoader(
        dataset=train_fold, batch_size=batch_size, shuffle=True, drop_last=False
    )
    test_loader = DataLoader(
        dataset=test_fold, batch_size=batch_size, shuffle=True, drop_last=False
    )
    return train_loader, test_loader


def train_test(
    i, train_loader,test_loader,batch_size, data_transform, model, optimizer, lr_scheduler, criterion, device, writer,record
):
    # kf = KFold(n_splits=10, shuffle=True, random_state=random.randint(0, 10))
    
    num = 1
    
    train_acculist, train_F1list, train_losslist,test_acculist, test_F1list, test_losslist = record
    # for train_index, test_index in kf.split(data):
    #     train_loader, test_loader = get_train_test_dataloader(
    #         data, batch_size, train_index, test_index, data_transform
    #     )
    for num in range(10):
        train_loss = 0
        train_TP_TN_FP_FN=[0,0,0,0]
        test_loss = 0
        test_TP_TN_FP_FN=[0,0,0,0]
        batch_num = 1
        # 验证
        print("验证开始.....")
        model.eval()
        with torch.no_grad():
            for test_data in test_loader:
                images, labels = test_data
                images=[image.to(device) for image in images]
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                get_num_correct(outputs, labels,test_TP_TN_FP_FN)
       
        # 开始进行训练
        print("训练开始.....")
        model.train()
        for batch in train_loader:
            # if batch_num==2:
            #     break
            images, labels = batch
            images=[image.to(device) for image in images]
            labels = labels.to(device)
            preds = model(images)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            get_num_correct(preds, labels,train_TP_TN_FP_FN)
            batch_num += 1
        lr_scheduler.step()

        train_size,test_size=sum(train_TP_TN_FP_FN),sum(test_TP_TN_FP_FN)
        train_acc,train_F1 = get_acc_F1(train_TP_TN_FP_FN)
        test_acc,test_F1 = get_acc_F1(test_TP_TN_FP_FN)
        print(
            "epoch {} num {}: \ntrain_acc: {:.2f}%\ntrain_F1: {:.2f}%\ntrain_loss: {:.2f}\ntest_acc: {:.2f}%\ntest_F1: {:.2f}%\ntest_loss: {:.2f}\n ".format(
                i,
                num,
                train_acc * 100,
                train_F1*100,
                train_loss / (train_size+esp),
                test_acc * 100,
                test_F1*100,
                test_loss / (test_size+esp),
            )
        )
        train_acculist.append(train_acc * 100)
        train_F1list.append(train_F1*100)
        train_losslist.append(train_loss/(train_size+esp))
        test_acculist.append(test_acc * 100)
        test_F1list.append(test_F1*100)
        test_losslist.append(test_loss/(test_size+esp))
        # writer.add_scalar("train_accu", train_correct / train_size * 100, i * 10 + num)
        # writer.add_scalar("train_loss", train_loss/train_size, i * 10 + num)
        # writer.add_scalar("test_accu", test_correct / test_size * 100, i * 10 + num)
        # writer.add_scalar("test_loss", test_loss/test_size, i * 10 + num)
        num += 1


if __name__ == "__main__":
    root_dir = r"/media/codelearner/E2EE175BEE1726F7/Users/QuickLearner/Documents/python/graduationProject/data"

    # 准备数据
    data = get_data(root_dir)
    # 数据转换器
    data_transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize((128, 128)), transforms.ToTensor()]
    )
    # 可视化writer
    writer = SummaryWriter("logs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 定义模型
    model = MultCNN(device).to(device)
    # 定义优化器和误差函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    criterion = nn.CrossEntropyLoss().to(device)

    
    epoch = 10
    for i in range(epoch):
        train_test(
            i,
            data,
            4,
            data_transform,
            model,
            optimizer,
            lr_scheduler,
            criterion,
            device,
            writer,
        )

    writer.close()
