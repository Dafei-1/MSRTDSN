import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


class main1(nn.Module):
    def __init__(self):
        super(main1, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 10, kernel_size=33, stride=16, padding=32),
            nn.BatchNorm1d(10),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(0.3)

        self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.avg2 = nn.AvgPool1d(kernel_size=3, stride=3)

    def forward(self, x):
        x = self.conv1(x)
        # print(f'Conv1 Output Shape: {x.shape}')
        x1 = self.dropout(x)
        # print(f'Conv2 Output Shape: {x1.shape}')
        x2 = self.avg1(x)
        # print(f'Conv3 Output Shape: {x2.shape}')
        x3 = self.avg2(x)
        # print(f'Conv3 Output Shape: {x3.shape}')

        # 使用插值调整 x2 和 x3 的尺寸到与 x1 一致
        x2 = F.interpolate(x2, size=x1.shape[2], mode='linear', align_corners=False)
        # print(f'Conv3 Output Shape: {x2.shape}')
        x3 = F.interpolate(x3, size=x1.shape[2], mode='linear', align_corners=False)
        # print(f'Conv3 Output Shape: {x3.shape}')
        x = torch.cat((x1, x2, x3), dim=1)
        # print(f'Conv1 Output Shape: {x.shape}')
        return x


#     64,30,10

class son1(nn.Module):
    def __init__(self):
        super(son1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(30, 60, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm1d(60),
            nn.ReLU()
        )
        self.pool = nn.MaxPool1d(4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        # print(f'Conv1 Output Shape: {x.shape}')
        return x


# 64,60,3

class main2(nn.Module):
    def __init__(self):
        super(main2, self).__init__()
        self.pool = nn.MaxPool1d(4)

        self.conv1 = nn.Sequential(
            nn.Conv1d(30, 20, kernel_size=9, stride=1, padding=1),
            nn.BatchNorm1d(20),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(0.3)

        self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.avg2 = nn.AvgPool1d(kernel_size=3, stride=3)

    def forward(self, x):
        x = self.conv1(x)
        # print(f'Conv1 Output Shape: {x.shape}')
        x1 = self.dropout(x)
        # print(f'Conv2 Output Shape: {x1.shape}')
        x2 = self.avg1(x)
        # print(f'Conv3 Output Shape: {x2.shape}')
        x3 = self.avg2(x)
        # print(f'Conv3 Output Shape: {x3.shape}')

        # 使用插值调整 x2 和 x3 的尺寸到与 x1 一致
        x2 = F.interpolate(x2, size=x1.shape[2], mode='linear', align_corners=False)
        # print(f'Conv3 Output Shape: {x2.shape}')
        x3 = F.interpolate(x3, size=x1.shape[2], mode='linear', align_corners=False)
        # print(f'Conv3 Output Shape: {x3.shape}')
        x = torch.cat((x1, x2, x3), dim=1)
        # print(f'Conv1 Output Shape: {x.shape}')
        return x


# 64,60,4

class main3(nn.Module):
    def __init__(self):
        super(main3, self).__init__()
        self.pool = nn.MaxPool1d(4)

        self.conv1 = nn.Sequential(
            nn.Conv1d(60, 30, kernel_size=6, stride=1, padding=1),
            nn.BatchNorm1d(30),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(0.3)

        self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.avg2 = nn.AvgPool1d(kernel_size=3, stride=3)

    def forward(self, x):
        x = self.conv1(x)
        # print(f'Conv1 Output Shape: {x.shape}')
        x1 = self.dropout(x)
        # print(f'Conv2 Output Shape: {x1.shape}')
        x2 = self.avg1(x)
        # print(f'Conv3 Output Shape: {x2.shape}')
        x3 = self.avg2(x)
        # print(f'Conv3 Output Shape: {x3.shape}')

        # 使用插值调整 x2 和 x3 的尺寸到与 x1 一致
        x2 = F.interpolate(x2, size=x1.shape[2], mode='linear', align_corners=False)
        # print(f'Conv3 Output Shape: {x2.shape}')
        x3 = F.interpolate(x3, size=x1.shape[2], mode='linear', align_corners=False)
        # print(f'Conv3 Output Shape: {x3.shape}')
        x = torch.cat((x1, x2, x3), dim=1)
        # print(f'Conv1 Output Shape: {x.shape}')
        return x


class main4(nn.Module):
    def __init__(self):
        super(main4, self).__init__()
        self.pool = nn.MaxPool1d(4)

        self.conv1 = nn.Sequential(
            nn.Conv1d(90, 40, kernel_size=6, stride=1, padding=1),
            nn.BatchNorm1d(40),
            nn.ReLU()
        )

        self.dropout = nn.Dropout(0.3)

        self.avg1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.avg2 = nn.AvgPool1d(kernel_size=3, stride=3)

    def forward(self, x):
        x = self.conv1(x)
        # print(f'Conv1 Output Shape: {x.shape}')
        x1 = self.dropout(x)
        # print(f'Conv2 Output Shape: {x1.shape}')
        x2 = self.avg1(x)
        # print(f'Conv3 Output Shape: {x2.shape}')
        x3 = self.avg2(x)
        # print(f'Conv3 Output Shape: {x3.shape}')

        # 使用插值调整 x2 和 x3 的尺寸到与 x1 一致
        x2 = F.interpolate(x2, size=x1.shape[2], mode='linear', align_corners=False)
        # print(f'Conv3 Output Shape: {x2.shape}')
        x3 = F.interpolate(x3, size=x1.shape[2], mode='linear', align_corners=False)
        # print(f'Conv3 Output Shape: {x3.shape}')
        x = torch.cat((x1, x2, x3), dim=1)
        # print(f'Conv1 Output Shape: {x.shape}')
        return x


class son2(nn.Module):
    def __init__(self):
        super(son2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(60, 90, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm1d(90),
            nn.ReLU()
        )
        self.pool = nn.MaxPool1d(4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        # print(f'Conv1 Output Shape: {x.shape}')
        return x


class son3(nn.Module):
    def __init__(self):
        super(son3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(90, 120, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm1d(120),
            nn.ReLU()
        )
        self.pool = nn.MaxPool1d(4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        # print(f'Conv1 Output Shape: {x.shape}')
        return x


class son4(nn.Module):
    def __init__(self):
        super(son4, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(120, 120, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm1d(120),
            nn.ReLU()
        )
        self.pool = nn.MaxPool1d(4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        # print(f'Conv1 Output Shape: {x.shape}')
        return x


class CFM(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CFM, self).__init__()

        # 定义激活函数
        act_fn = nn.ReLU(inplace=True)

        # 定义两个1D卷积层，输入通道数和输出通道数都为out_dim，kernel_size=3, stride=1, padding=1
        self.layer_10 = nn.Conv1d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_20 = nn.Conv1d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)

        # 定义两个1D卷积层接BatchNorm1d和ReLU激活函数的组合
        self.layer_11 = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_dim),
            act_fn
        )
        self.layer_21 = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_dim),
            act_fn
        )

        # 定义一个融合后的卷积层
        self.layer_ful1 = nn.Sequential(
            nn.Conv1d(out_dim * 2, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_dim),
            act_fn
        )

    def forward(self, x1, x2):
        # 通过两个卷积层分别处理x1和x2
        x_rgb = self.layer_10(x1)
        x2 = F.interpolate(x2, size=x1.shape[2], mode='linear', align_corners=False)
        x_dep = self.layer_20(x2)
        # print(x1.shape)  # 输出 x1 的形状
        # print(x2.shape)  # 输出 dep_w 的形状

        # 使用Sigmoid函数生成权重
        rgb_w = torch.sigmoid(x_rgb)
        dep_w = torch.sigmoid(x_dep)

        # 对输入的x1和x2进行加权
        x_rgb_w = x1 * dep_w
        x_dep_w = x2 * rgb_w

        # 对加权后的特征进行残差计算
        x_rgb_r = x_rgb_w + x1
        x_dep_r = x_dep_w + x2

        # 分别通过卷积层进行处理
        x_rgb_r = self.layer_11(x_rgb_r)
        x_dep_r = self.layer_21(x_dep_r)

        # 将两个处理后的输出在通道维度上进行拼接
        ful_out = torch.cat((x_rgb_r, x_dep_r), dim=1)

        # 最后的融合卷积层
        out1 = self.layer_ful1(ful_out)

        return out1


class MSTACNNc(nn.Module):
    def __init__(self):
        super(MSTACNNc, self).__init__()
        self.m1 = main1()
        self.m2 = main2()
        self.m3 = main3()
        self.m4 = main4()

        self.s1_1 = son1()
        self.s1_2 = son1()
        self.s1_3 = son1()

        self.s2_1 = son2()
        self.s2_2 = son2()
        self.s2_3 = son2()

        self.s3_1 = son3()
        self.s3_2 = son3()
        self.s3_3 = son3()

        self.s4_1 = son4()
        self.s4_2 = son4()
        self.s4_3 = son4()

        self.c1 = CFM(60, 60)
        self.c2 = CFM(90, 90)
        self.c3 = CFM(120, 120)

        self.task1_fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            # nn.Dropout(0.2),
            # nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(120, 4)
        )

        self.task2_fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            # nn.Dropout(0.2),
            # nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(120, 4)
        )

        self.task3_fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            # nn.Dropout(0.2),
            # nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(120, 3)
        )

    def forward(self, x):
        x = self.m1(x)
        x1 = self.s1_1(x)
        x2 = self.s1_2(x)
        x3 = self.s1_3(x)

        x = self.m2(x)
        x1 = self.c1(x, x1)
        x1 = self.s2_1(x1)
        x2 = self.c1(x, x2)
        x2 = self.s2_2(x2)
        x3 = self.c1(x, x3)
        x3 = self.s2_3(x3)

        x = self.m3(x)
        x1 = self.c2(x, x1)
        x1 = self.s3_1(x1)
        x2 = self.c2(x, x2)
        x2 = self.s3_2(x2)
        x3 = self.c2(x, x3)
        x3 = self.s3_3(x3)

        x = self.m4(x)
        x1 = self.c3(x, x1)
        x1 = self.s4_1(x1)
        x2 = self.c3(x, x2)
        x2 = self.s4_2(x2)
        x3 = self.c3(x, x3)
        x3 = self.s4_3(x3)
        # print('x1shap', x1.shape)
        # print('x2shap', x2.shape)
        # print('x3shap', x3.shape)

        t1 = self.task1_fc(x1)
        t2 = self.task2_fc(x2)
        t3 = self.task3_fc(x3)

        return t1, t2, t3


class MSTACNNp(nn.Module):
    def __init__(self):
        super(MSTACNNp, self).__init__()
        self.m1 = main1()
        self.m2 = main2()
        self.m3 = main3()
        self.m4 = main4()

        self.s1_1 = son1()
        self.s1_2 = son1()
        self.s1_3 = son1()

        self.s2_1 = son2()
        self.s2_2 = son2()
        self.s2_3 = son2()

        self.s3_1 = son3()
        self.s3_2 = son3()
        self.s3_3 = son3()

        self.s4_1 = son4()
        self.s4_2 = son4()
        self.s4_3 = son4()

        self.c1 = CFM(60, 60)
        self.c2 = CFM(90, 90)
        self.c3 = CFM(120, 120)

        self.task1_fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            # nn.Dropout(0.2),
            # nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(120, 10)
        )

        self.task2_fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            # nn.Dropout(0.2),
            # nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(120, 4)
        )

        self.task3_fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            # nn.Dropout(0.2),
            # nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(120, 3)
        )

    def forward(self, x):
        x = self.m1(x)
        x1 = self.s1_1(x)
        x2 = self.s1_2(x)
        x3 = self.s1_3(x)

        x = self.m2(x)
        x1 = self.c1(x, x1)
        x1 = self.s2_1(x1)
        x2 = self.c1(x, x2)
        x2 = self.s2_2(x2)
        x3 = self.c1(x, x3)
        x3 = self.s2_3(x3)

        x = self.m3(x)
        x1 = self.c2(x, x1)
        x1 = self.s3_1(x1)
        x2 = self.c2(x, x2)
        x2 = self.s3_2(x2)
        x3 = self.c2(x, x3)
        x3 = self.s3_3(x3)

        x = self.m4(x)
        x1 = self.c3(x, x1)
        x1 = self.s4_1(x1)
        x2 = self.c3(x, x2)
        x2 = self.s4_2(x2)
        x3 = self.c3(x, x3)
        x3 = self.s4_3(x3)
        # print('x1shap', x1.shape)
        # print('x2shap', x2.shape)
        # print('x3shap', x3.shape)

        t1 = self.task1_fc(x1)
        t2 = self.task2_fc(x2)
        t3 = self.task3_fc(x3)

        return t1, t2, t3


class MSTACNNct(nn.Module):
    def __init__(self):
        super(MSTACNNct, self).__init__()
        self.m1 = main1()
        self.m2 = main2()
        self.m3 = main3()
        self.m4 = main4()

        self.s1_1 = son1()
        self.s1_2 = son1()
        self.s1_3 = son1()

        self.s2_1 = son2()
        self.s2_2 = son2()
        self.s2_3 = son2()

        self.s3_1 = son3()
        self.s3_2 = son3()
        self.s3_3 = son3()

        self.s4_1 = son4()
        self.s4_2 = son4()
        self.s4_3 = son4()

        self.c1 = CFM(60, 60)
        self.c2 = CFM(90, 90)
        self.c3 = CFM(120, 120)

        self.task1_fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(120, 4)
        )

        self.task2_fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(120, 4)
        )

        self.task3_fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(120, 3)
        )

    def forward(self, x, return_features=False):
        # Initialize a dictionary to store features
        outputs = {}

        # First layer - main1
        x = self.m1(x)
        x1 = self.s1_1(x)
        x2 = self.s1_2(x)
        x3 = self.s1_3(x)

        # Store features for main1 layer
        if return_features:
            outputs['x1'] = x1
            outputs['x2'] = x2
            outputs['x3'] = x3

        # Second layer - main2
        x = self.m2(x)
        x1_c1 = self.c1(x, x1)
        x2_c1 = self.c1(x, x2)
        x3_c1 = self.c1(x, x3)
        x1_c1 = self.s2_1(x1_c1)
        x2_c1 = self.s2_2(x2_c1)
        x3_c1 = self.s2_3(x3_c1)

        # Store features for C1 layer
        if return_features:
            outputs['C1_x1'] = x1_c1
            outputs['C1_x2'] = x2_c1
            outputs['C1_x3'] = x3_c1

        # Third layer - main3
        x = self.m3(x)
        x1_c2 = self.c2(x, x1_c1)
        x2_c2 = self.c2(x, x2_c1)
        x3_c2 = self.c2(x, x3_c1)
        x1_c2 = self.s3_1(x1_c2)
        x2_c2 = self.s3_2(x2_c2)
        x3_c2 = self.s3_3(x3_c2)

        # Store features for C2 layer
        if return_features:
            outputs['C2_x1'] = x1_c2
            outputs['C2_x2'] = x2_c2
            outputs['C2_x3'] = x3_c2

        # Fourth layer - main4
        x = self.m4(x)
        x1_c3 = self.c3(x, x1_c2)
        x2_c3 = self.c3(x, x2_c2)
        x3_c3 = self.c3(x, x3_c2)
        x1_c3 = self.s4_1(x1_c3)
        x2_c3 = self.s4_2(x2_c3)
        x3_c3 = self.s4_3(x3_c3)

        # Store features for C3 layer
        if return_features:
            outputs['C3_x1'] = x1_c3
            outputs['C3_x2'] = x2_c3
            outputs['C3_x3'] = x3_c3

        # Task-specific final layers
        t1 = self.task1_fc(x1_c3)
        t2 = self.task2_fc(x2_c3)
        t3 = self.task3_fc(x3_c3)

        # Store features for FC layer
        if return_features:
            outputs['FC_x1'] = t1
            outputs['FC_x2'] = t2
            outputs['FC_x3'] = t3

        return outputs if return_features else (t1, t2, t3)


class MSTACNNpt(nn.Module):
    def __init__(self):
        super(MSTACNNpt, self).__init__()
        self.m1 = main1()
        self.m2 = main2()
        self.m3 = main3()
        self.m4 = main4()

        self.s1_1 = son1()
        self.s1_2 = son1()
        self.s1_3 = son1()

        self.s2_1 = son2()
        self.s2_2 = son2()
        self.s2_3 = son2()

        self.s3_1 = son3()
        self.s3_2 = son3()
        self.s3_3 = son3()

        self.s4_1 = son4()
        self.s4_2 = son4()
        self.s4_3 = son4()

        self.c1 = CFM(60, 60)
        self.c2 = CFM(90, 90)
        self.c3 = CFM(120, 120)

        self.task1_fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(120, 10)
        )

        self.task2_fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(120, 4)
        )

        self.task3_fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(120, 3)
        )

    def forward(self, x, return_features=False):
        # Initialize a dictionary to store features
        outputs = {}

        # First layer - main1
        x = self.m1(x)
        x1 = self.s1_1(x)
        x2 = self.s1_2(x)
        x3 = self.s1_3(x)

        # Store features for main1 layer
        if return_features:
            outputs['x1'] = x1
            outputs['x2'] = x2
            outputs['x3'] = x3

        # Second layer - main2
        x = self.m2(x)
        x1_c1 = self.c1(x, x1)
        x2_c1 = self.c1(x, x2)
        x3_c1 = self.c1(x, x3)
        x1_c1 = self.s2_1(x1_c1)
        x2_c1 = self.s2_2(x2_c1)
        x3_c1 = self.s2_3(x3_c1)

        # Store features for C1 layer
        if return_features:
            outputs['C1_x1'] = x1_c1
            outputs['C1_x2'] = x2_c1
            outputs['C1_x3'] = x3_c1

        # Third layer - main3
        x = self.m3(x)
        x1_c2 = self.c2(x, x1_c1)
        x2_c2 = self.c2(x, x2_c1)
        x3_c2 = self.c2(x, x3_c1)
        x1_c2 = self.s3_1(x1_c2)
        x2_c2 = self.s3_2(x2_c2)
        x3_c2 = self.s3_3(x3_c2)

        # Store features for C2 layer
        if return_features:
            outputs['C2_x1'] = x1_c2
            outputs['C2_x2'] = x2_c2
            outputs['C2_x3'] = x3_c2

        # Fourth layer - main4
        x = self.m4(x)
        x1_c3 = self.c3(x, x1_c2)
        x2_c3 = self.c3(x, x2_c2)
        x3_c3 = self.c3(x, x3_c2)
        x1_c3 = self.s4_1(x1_c3)
        x2_c3 = self.s4_2(x2_c3)
        x3_c3 = self.s4_3(x3_c3)

        # Store features for C3 layer
        if return_features:
            outputs['C3_x1'] = x1_c3
            outputs['C3_x2'] = x2_c3
            outputs['C3_x3'] = x3_c3

        # Task-specific final layers
        t1 = self.task1_fc(x1_c3)
        t2 = self.task2_fc(x2_c3)
        t3 = self.task3_fc(x3_c3)

        # Store features for FC layer
        if return_features:
            outputs['FC_x1'] = t1
            outputs['FC_x2'] = t2
            outputs['FC_x3'] = t3

        return outputs if return_features else (t1, t2, t3)


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
    Net = MSTACNNc().to(device)

    # 前向传播，计算输出
    output = Net(t)

    # 打印输出
    print(output)

    # 打印模型结构
    print(Net)
