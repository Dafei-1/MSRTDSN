import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class WDCNN1(nn.Module):
    def __init__(self):
        super(WDCNN1, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=8, padding=32),  # 'same' padding in PyTorch
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU()
        )

        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)

        self.dense = nn.Linear(64, 109)  # Assuming you have 109 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        # print(x.shape)
        x = self.global_avg_pooling(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)  # Add the third dimension
        # print(x.shape)
        x = self.dense(x)
        # print(x.shape)

        return x


class WDCNNC(nn.Module):
    def __init__(self):
        super(WDCNNC, self).__init__()
        self.class_1 = 4
        self.class_2 = 4
        self.class_3 = 3
        self.encoder = WDCNN1()  # Use WDCNN as the backbone
        self.size_fc = nn.Sequential(
            # nn.AdaptiveAvgPool1d(1),
            Flatten(),
            nn.Linear(109, 128),
            nn.ReLU(),
            nn.Linear(128, self.class_1)
        )
        self.type_fc = nn.Sequential(
            # nn.AdaptiveAvgPool1d(1),
            Flatten(),
            nn.Linear(109, 128),
            nn.ReLU(),
            nn.Linear(128, self.class_2)
        )
        self.load_fc = nn.Sequential(
            # nn.AdaptiveAvgPool1d(1),
            Flatten(),
            nn.Linear(109, 128),
            nn.ReLU(),
            nn.Linear(128, self.class_3)
        )

    def forward(self, x):
        x = self.encoder(x)
        x1 = self.size_fc(x)
        x2 = self.type_fc(x)
        x3 = self.load_fc(x)
        return x1, x2, x3


class WDCNNP(nn.Module):
    def __init__(self):
        super(WDCNNP, self).__init__()
        self.class_1 = 10
        self.class_2 = 4
        self.class_3 = 3
        self.encoder = WDCNN1()  # Use WDCNN as the backbone
        self.size_fc = nn.Sequential(
            Flatten(),
            nn.Linear(109, 128),
            nn.ReLU(),
            nn.Linear(128, self.class_1)
        )
        self.type_fc = nn.Sequential(
            Flatten(),
            nn.Linear(109, 128),
            nn.ReLU(),
            nn.Linear(128, self.class_2)
        )
        self.load_fc = nn.Sequential(
            Flatten(),
            nn.Linear(109, 128),
            nn.ReLU(),
            nn.Linear(128, self.class_3)
        )

    def forward(self, x):
        x = self.encoder(x)
        x1 = self.size_fc(x)
        x2 = self.type_fc(x)
        x3 = self.load_fc(x)
        return x1, x2, x3


if __name__ == '__main__':
    # t = torch.randn(32, 3, 300, 300)
    # Net = Net1()
    # output = Net(t)
    # print(output)
    # print(Net)
    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 生成随机输入数据 (batch_size=32, channels=3, height=300, width=300)
    t = torch.randn(100, 1, 1024).to(device)  # 将输入数据迁移到 GPU

    # 实例化模型，并迁移到 GPU
    Net = WDCNNC().to(device)

    # 前向传播，计算输出
    output = Net(t)

    # 打印输出
    print(output)

    # 打印模型结构
    print(Net)
