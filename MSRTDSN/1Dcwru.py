import csv
from datetime import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import torch.optim as optim
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from NET import Net11, Net12, Net13, Net14
from RES2NET import res2netc
import matplotlib.pyplot as plt
from MSTACNN import MSTACNNc, MSTACNNct
from WDCNN import WDCNNC
from MATGN import MTAGNc
from sklearn.manifold import TSNE

from OLSR import OLSR

root_path = "E:\\1ex2\\cwru"
data_file_path = os.path.join(root_path, "cwru_data.csv")
label_file_path = os.path.join(root_path, "cwru_label.csv")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dataset(Dataset):
    def __init__(self, data_path, label_path, transform=None):
        """
        Args:
            data_path (string): 时间序列数据 CSV 文件路径。
            label_path (string): 标签 CSV 文件路径，包含多个任务的标签。
            transform (callable, optional): 可选的转换函数，默认为 None。
        """
        self.data = pd.read_csv(data_path, header=None)  # 读取时间序列数据
        self.labels = pd.read_csv(label_path, header=None)  # 读取标签数据
        self.transform = transform

    def __len__(self):
        return len(self.data)  # 返回样本数量

    def __getitem__(self, idx):
        # 获取对应的时间序列数据
        seq_data = self.data.iloc[idx].values.astype(np.float32)  # 获取时间序列数据并转换为浮点数
        labels = self.labels.iloc[idx].values.astype(np.int64)  # 获取多个标签并转换为整型

        # 转换为 Tensor
        seq_tensor = torch.tensor(seq_data).unsqueeze(0)  # 将时间序列数据转换为 Tensor
        # print(f"Data shape: {seq_tensor.shape}")

        # 如果有 transform 操作，应用于序列数据
        if self.transform:
            seq_tensor = self.transform(seq_tensor)

        # 返回时间序列数据和多个任务的标签
        # 假设标签有三个任务：task1, task2, task3
        task1_label = labels[0]
        task2_label = labels[1]
        task3_label = labels[2]

        return seq_tensor, (task1_label, task2_label, task3_label)  # 返回时间序列数据和三个标签


# 加载数据集
dataset = Dataset(data_file_path, label_file_path)

# 获取所有的索引
indices = list(range(len(dataset)))

# 按比例分割数据集，80% 训练集，20% 测试集
train_indices, test_indices = train_test_split(indices, test_size=0.7, random_state=42)

# 使用 Subset 来创建训练集和测试集
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")

class Trainer:
    def __init__(self, model, train_dataset, test_dataset):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def train(self, train_loader_, test_loader_, optimizer_, scheduler_, epochs=100):
        # loss_func = torch.nn.CrossEntropyLoss()
        loss_func = OLSR()
        train_batch = len(train_loader_)
        valid_batch = len(test_loader_)

        print('--------------------Training--------------------')

        avg_cost = np.zeros([epochs, 8])  # 0/1/2/3 训练损失; 4/5/6/7 验证损失

        # 用于存储准确率
        task1_acc_history = []
        task2_acc_history = []
        task3_acc_history = []
        avg_acc_history = []

        # 使用固定的权重
        fixed_weight = 1.0  # 可以设置为你想要的固定值

        for epoch in range(epochs):
            # 训练阶段
            train_loss_1 = 0.0
            train_loss_2 = 0.0
            train_loss_3 = 0.0
            train_loss = 0.0
            train_acc_1 = 0
            train_acc_2 = 0
            train_acc_3 = 0

            self.model.train()
            for batch_idx, data in enumerate(train_loader_):
                optimizer_.zero_grad()
                inputs, (label1, label2, label3) = data
                inputs, label1, label2, label3 = inputs.to(device), label1.to(device), label2.to(device), label3.to(
                    device)

                # 前向传播
                output1, output2, output3 = self.model(inputs)

                # 计算任务损失
                tr_loss_1 = loss_func(output1, label1)
                tr_loss_2 = loss_func(output2, label2)
                tr_loss_3 = loss_func(output3, label3)

                # 使用固定的权重进行加权
                loss = fixed_weight * (tr_loss_1 + tr_loss_2 + tr_loss_3)

                # 累加训练损失
                train_loss_1 += tr_loss_1.item()
                train_loss_2 += tr_loss_2.item()
                train_loss_3 += tr_loss_3.item()
                train_loss += loss.item()

                loss.backward()
                optimizer_.step()

                # 计算任务的准确率
                train_acc_1 += np.sum(np.argmax(output1.cpu().data.numpy(), axis=1) == label1.cpu().numpy())
                train_acc_2 += np.sum(np.argmax(output2.cpu().data.numpy(), axis=1) == label2.cpu().numpy())
                train_acc_3 += np.sum(np.argmax(output3.cpu().data.numpy(), axis=1) == label3.cpu().numpy())

            # 计算训练准确率
            train_acc_1 = (train_acc_1 / len(self.train_dataset)) * 100
            train_acc_2 = (train_acc_2 / len(self.train_dataset)) * 100
            train_acc_3 = (train_acc_3 / len(self.train_dataset)) * 100

            task1_acc_history.append(train_acc_1)
            task2_acc_history.append(train_acc_2)
            task3_acc_history.append(train_acc_3)
            avg_acc_history.append((train_acc_1 + train_acc_2 + train_acc_3) / 3)

            # 验证阶段
            self.model.eval()
            with torch.no_grad():
                test_loss_1 = 0.0
                test_loss_2 = 0.0
                test_loss_3 = 0.0
                test_loss = 0.0
                test_acc_1 = 0
                test_acc_2 = 0
                test_acc_3 = 0

                for data in test_loader_:
                    inputs, (label1, label2, label3) = data
                    inputs, label1, label2, label3 = inputs.to(device), label1.to(device), label2.to(device), label3.to(
                        device)

                    # 前向传播
                    output1, output2, output3 = self.model(inputs)

                    # 计算验证损失
                    test_loss_1 = loss_func(output1, label1)
                    test_loss_2 = loss_func(output2, label2)
                    test_loss_3 = loss_func(output3, label3)
                    loss = fixed_weight * (test_loss_1 + test_loss_2 + test_loss_3)

                    # 累加验证损失
                    test_loss_1 += test_loss_1.item()
                    test_loss_2 += test_loss_2.item()
                    test_loss_3 += test_loss_3.item()
                    test_loss += loss.item()

                    # 计算任务的验证准确率
                    test_acc_1 += np.sum(np.argmax(output1.cpu().data.numpy(), axis=1) == label1.cpu().numpy())
                    test_acc_2 += np.sum(np.argmax(output2.cpu().data.numpy(), axis=1) == label2.cpu().numpy())
                    test_acc_3 += np.sum(np.argmax(output3.cpu().data.numpy(), axis=1) == label3.cpu().numpy())

                # 计算验证准确率
                test_acc_1 = 100 * test_acc_1 / len(self.test_dataset)
                test_acc_2 = 100 * test_acc_2 / len(self.test_dataset)
                test_acc_3 = 100 * test_acc_3 / len(self.test_dataset)

            scheduler_.step()  # 更新学习率

            # 计算训练和验证损失的平均值
            avg_train_loss = train_loss / len(train_loader_)
            avg_valid_loss = test_loss / len(test_loader_)

            # 打印结果
            print(f"Epoch [{epoch + 1}/{epochs}] | "
                  f"Train Loss: {avg_train_loss:.5f} | "
                  f"Train FFI_acc: {train_acc_1:.2f}% | "
                  f"Train FSI_acc: {train_acc_2:.2f}% | "
                  f"Train FTI_acc: {train_acc_3:.2f}% | "
                  f"Valid Loss: {avg_valid_loss:.5f} | "
                  f"Valid FFI_acc: {test_acc_1:.2f}% | "
                  f"Valid FSI_acc: {test_acc_2:.2f}% | "
                  f"Valid FTI_acc: {test_acc_3:.2f}%")

        print('训练完成！')


class Trainer2:
    def __init__(self, model, train_dataset, test_dataset):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def train(self, train_loader_, test_loader_, optimizer_, scheduler_, epochs=100):
        # loss_func = torch.nn.CrossEntropyLoss()
        loss_fuc = OLSR()
        train_batch = len(train_loader_)
        valid_batch = len(test_loader_)
        T = 2

        print('--------------------Training--------------------')

        avg_cost = np.zeros([epochs, 8])  # 0/1/2/3 训练损失; 4/5/6/7 验证损失

        # 用于存储准确率
        task1_acc_history = []
        task2_acc_history = []
        task3_acc_history = []
        avg_acc_history = []

        # RLW: 随机生成每个任务的权重
        lambda_weight = np.abs(np.random.normal(size=(3, epochs)))  # 为3个任务生成随机权重
        lambda_weight /= np.sum(lambda_weight, axis=0)  # 对权重进行归一化处理

        for epoch in range(epochs):
            # 训练阶段
            train_loss_1 = 0.0
            train_loss_2 = 0.0
            train_loss_3 = 0.0
            train_loss = 0.0
            train_acc_1 = 0
            train_acc_2 = 0
            train_acc_3 = 0

            cost = np.zeros(8, dtype=np.float32)

            if epoch > 1:
                # 动态调整任务权重
                w_1 = avg_cost[epoch - 1, 0] / avg_cost[epoch - 2, 0]
                w_2 = avg_cost[epoch - 1, 1] / avg_cost[epoch - 2, 1]
                w_3 = avg_cost[epoch - 1, 2] / avg_cost[epoch - 2, 2]
                total_weight = w_1 + w_2 + w_3
                lambda_weight[0, epoch] = 3 * w_1 / total_weight
                lambda_weight[1, epoch] = 3 * w_2 / total_weight
                lambda_weight[2, epoch] = 3 * w_3 / total_weight

            self.model.train()
            for batch_idx, data in enumerate(train_loader_):
                optimizer_.zero_grad()
                inputs, (label1, label2, label3) = data
                inputs, label1, label2, label3 = inputs.to(device), label1.to(device), label2.to(device), label3.to(
                    device)

                # 前向传播
                output1, output2, output3 = self.model(inputs)

                # 计算任务损失
                tr_loss = [loss_fuc(output1, label1),
                           loss_fuc(output2, label2),
                           loss_fuc(output3, label3)]

                # 动态权重调整
                loss = (lambda_weight[0, epoch] * tr_loss[0] +
                        lambda_weight[1, epoch] * tr_loss[1] +
                        lambda_weight[2, epoch] * tr_loss[2])

                # 累加每个任务的损失
                train_loss_1 += tr_loss[0].item()
                train_loss_2 += tr_loss[1].item()
                train_loss_3 += tr_loss[2].item()
                train_loss += loss.item()

                loss.backward()
                optimizer_.step()

                cost[0] = tr_loss[0].item()
                cost[1] = tr_loss[1].item()
                cost[2] = tr_loss[2].item()
                cost[3] = loss.item()

                avg_cost[epoch, :4] += cost[:4] / train_batch

                # 计算准确率
                train_acc_1 += np.sum(np.argmax(output1.cpu().data.numpy(), axis=1) == label1.cpu().numpy())
                train_acc_2 += np.sum(np.argmax(output2.cpu().data.numpy(), axis=1) == label2.cpu().numpy())
                train_acc_3 += np.sum(np.argmax(output3.cpu().data.numpy(), axis=1) == label3.cpu().numpy())

            train_acc_1 = (train_acc_1 / train_dataset.__len__()) * 100
            train_acc_2 = (train_acc_2 / train_dataset.__len__()) * 100
            train_acc_3 = (train_acc_3 / train_dataset.__len__()) * 100

            # 记录准确率
            task1_acc_history.append(train_acc_1)
            task2_acc_history.append(train_acc_2)
            task3_acc_history.append(train_acc_3)
            avg_acc_history.append((train_acc_1 + train_acc_2 + train_acc_3) / 3)

            # 验证阶段
            self.model.eval()
            with torch.no_grad():
                test_loss_1 = 0.0
                test_loss_2 = 0.0
                test_loss_3 = 0.0
                test_loss = 0.0
                test_acc_1 = 0
                test_acc_2 = 0
                test_acc_3 = 0

                if epoch + 1 == epochs:
                    pred_label1 = np.array([], dtype=np.int64)
                    pred_label2 = np.array([], dtype=np.int64)
                    pred_label3 = np.array([], dtype=np.int64)
                for data in test_loader_:
                    inputs, (label1, label2, label3) = data
                    inputs, label1, label2, label3 = inputs.to(device), label1.to(device), label2.to(device), label3.to(
                        device)
                    output1, output2, output3 = self.model(inputs)
                    val_loss = [loss_fuc(output1, label1),
                                loss_fuc(output2, label2),
                                loss_fuc(output3, label3)]
                    loss = sum(lambda_weight[i, epoch] * val_loss[i] for i in range(3))

                    test_loss_1 += val_loss[0].item()
                    test_loss_2 += val_loss[1].item()
                    test_loss_3 += val_loss[2].item()
                    test_loss += loss.item()

                    cost[4] = val_loss[0].item()
                    cost[5] = val_loss[1].item()
                    cost[6] = val_loss[2].item()
                    cost[7] = loss.item()

                    avg_cost[epoch, 4:8] += cost[4:8] / valid_batch

                    test_acc_1 += np.sum(np.argmax(output1.cpu().data.numpy(), axis=1) == label1.cpu().numpy())
                    test_acc_2 += np.sum(np.argmax(output2.cpu().data.numpy(), axis=1) == label2.cpu().numpy())
                    test_acc_3 += np.sum(np.argmax(output3.cpu().data.numpy(), axis=1) == label3.cpu().numpy())

                    if epoch + 1 == epochs:
                        pred_label1 = np.append(pred_label1, torch.max(output1, dim=1)[1].cpu().numpy().astype('int64'))
                        pred_label2 = np.append(pred_label2, torch.max(output2, dim=1)[1].cpu().numpy().astype('int64'))
                        pred_label3 = np.append(pred_label3, torch.max(output3, dim=1)[1].cpu().numpy().astype('int64'))

                test_acc_1 = 100 * test_acc_1 / test_dataset.__len__()
                test_acc_2 = 100 * test_acc_2 / test_dataset.__len__()
                test_acc_3 = 100 * test_acc_3 / test_dataset.__len__()

            scheduler_.step()

            avg_train_loss = (train_loss / len(train_loader_))
            avg_valid_loss = (test_loss / len(test_loader_))

            # 打印结果
            print(f"Epoch [{epoch + 1}/{epochs}] | "
                  f"Train Loss: {avg_train_loss:.5f} | "
                  f"Train FFI_acc: {train_acc_1:.2f}% | "
                  f"Train FSI_acc: {train_acc_2:.2f}% | "
                  f"Train FTI_acc: {train_acc_3:.2f}% | "
                  f"Valid Loss: {avg_valid_loss:.5f} | "
                  f"Valid FFI_acc: {test_acc_1:.2f}% | "
                  f"Valid FSI_acc: {test_acc_2:.2f}% | "
                  f"Valid FTI_acc: {test_acc_3:.2f}%")

        print('训练完成！')


if __name__ == '__main__':
    # ==================== Hyper parameters =====================
    EPOCHS = 100
    BATCH_SIZE = 128
    LR = 0.0001
    # set_seed(2021)

    # train_dataset = train_dataset
    # test_dataset = test_dataset
    # define model, vis, optimiser
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # vis = visdom.Visdom(env='dwa')
    # model = MSTACNNc().to(device)
    # model = MSTACNNct().to(device)
    # model = WDCNNC().to(device)
    model = MTAGNc().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # trainer无自适应，trainer1有自适应
    my_trainer = Trainer2(model, train_dataset, test_dataset)
    # my_trainer = trainer1(model)
    # my_trainer = trainer21(model)

    my_trainer.train(train_loader, test_loader, optimizer, scheduler, EPOCHS)
