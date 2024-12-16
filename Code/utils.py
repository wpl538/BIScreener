import cv2
import numpy as np   #科学计数库
import pandas as pd  #基于Numpy的Python数据分析工具，提供了高效的数据结构和数据分析工具
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score    #可以使用这两个函数比较回归模型性能，评估模型的拟合程度和预测精度

import torch #提供了许多深度学习任务的函数和类
from torch import nn, optim #nn用于定义神经网络层和模型,optim用于定义各种优化算法
from torch.utils.data import DataLoader, Dataset  #使用DataLoader, Dataset加载处理数据
import torchvision
from torchvision import datasets, transforms  #通过导入可以方便的访问和使用现有的计算机视觉数据集，并对数据进行预处理以满足模型训练的需求
from torchvision import models
from datetime import datetime
from tqdm import tqdm    #tqdm用于创建进度条的库，可以在循环中显示进度条，方便用户实时了解任务的完成情况
#import torchmetrics      #提供了多种用于评估和度量深度学习模型性能的指标
import matplotlib.pyplot as plt            #Python中的一个绘图库，提供了各种绘图函数和工具，可以用各种类型图形和可视化结果
import scipy.io as scio              #读取和写入MATLAB格式的数据文件（.mat文件）
from sklearn.metrics import classification_report
from itertools import cycle
from sklearn.preprocessing import label_binarize    #都属于sklearn库中metrics模块和preprocessing模块，用于评估二分类模型的性能
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import os
from torchvision.utils import save_image
import math
import shutil
import random
import glob
from PIL import Image
import itertools
from sklearn.metrics import confusion_matrix,roc_curve,auc
import torch.nn.functional as F
import seaborn as sns
import time

print(torch.cuda.is_available())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  #用于检查是否有可用的GPU，并根据结果设置设备

class CustomDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx, 0]
        label = self.df.iloc[idx, 1]
        image = Image.open(img_path)  # 确保已经导入PIL库

        if self.transform:
            image = self.transform(image)

        return image, label


def set_random_seed(seed: int):
    """设置全局随机种子以确保可重复性。

    Args:
        seed (int): 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子 {seed} 已设置")

def initialize_model_fc(model, num_classes, learning_rate, device, last_layer_name=None):
        """初始化模型的最后一个全连接层。

        Args:
            model (nn.Module): 预训练的模型
            num_classes (int): 新分类任务的类别数
            learning_rate (float): 新全连接层的学习率
            device (torch.device): 训练设备 (e.g., 'cuda' or 'cpu')
            last_layer_name (str, optional): 最后一层的名称。如果为 None，自动识别
        """
        # 确保模型在指定设备上
        model.to(device)

        if last_layer_name is None:
            # 自动识别最后一层
            last_layer_name = list(model._modules.keys())[-1]

        # 获取最后一层
        last_layer = getattr(model, last_layer_name)

        if isinstance(last_layer, nn.Linear):
            # 替换最后一个全连接层
            num_features = last_layer.in_features
            model.__setattr__(last_layer_name, nn.Linear(num_features, num_classes).to(device))
        else:
            raise ValueError("最后一层不是 nn.Linear 类型，无法替换")

        #print(model)

        # 设置新全连接层的学习率
        new_fc_layer = getattr(model, last_layer_name)
        for param in new_fc_layer.parameters():
            param.requires_grad = True
            if param.dim() > 1:  # 权重参数
                nn.init.xavier_uniform_(param)
                param.lr = learning_rate
            else:  # 偏置参数
                nn.init.constant_(param, 0)
                param.lr = learning_rate

        return model

class EarlyStopping:
    def __init__(self, patience=100, delta=0, verbose=10, path='checkpoint.pt', trace_func=print):
        self.patience = patience  # 容忍的等待次数
        self.delta = delta  # 验证集上的性能提升的阈值
        self.verbose = verbose  # 是否输出提示信息
        self.path = path  # 保存模型的路径
        self.trace_func = trace_func  # 输出信息的函数
        self.counter = 0  # 等待次数计数
        self.best_score = None  # 最好的验证集性能
        self.early_stop = False  # 是否早停
        self.val_loss_min = np.Inf  # 最小的验证集损失

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_with_early_stopping(model, train_loader, criterion, optimizer, num_epochs, valid_loader, early_stopper=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_list = []
    val_loss_list = []
    accuracy_list = []
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        sum_loss = 0.0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()

        epoch_loss = sum_loss / len(train_loader)
        loss_list.append(epoch_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # Validate model performance
        if valid_loader is not None:
            model.eval()
            val_sum_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for imgs, labels in valid_loader:
                    imgs = imgs.to(device)
                    labels = labels.to(device)
                    out = model(imgs)
                    val_loss = criterion(out, labels).item()
                    _, predicted = torch.max(out.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                val_loss /= len(valid_loader)
                val_loss_list.append(val_loss)
                accuracy = 100 * correct / total
                accuracy_list.append(accuracy)
                print('Validation Loss: {:.4f}, Validation Accuracy: {:.2f}'.format(val_loss, accuracy))

                # Check for early stopping using validation accuracy
                if early_stopper:
                    early_stopper(val_loss, model)
                    if early_stopper.early_stop:
                        print("Early stopping")
                        break

                        # Save the best model based on validation accuracy
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_state = model.state_dict()

                    # Load the best model state
    if 'best_model_state' in locals():
        model.load_state_dict(best_model_state)

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 11

    # Plot the training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(loss_list)), loss_list, "r-", label="Training Loss")
    plt.plot(range(len(val_loss_list)), val_loss_list, "b-", label="Validation Loss")
    plt.xlabel("Iterations (x100)", fontweight='bold')
    plt.ylabel("Loss", fontweight='bold')
    plt.legend()
   # plt.title("Training and Validation Loss")
    plt.grid(True)
    plt.show()

    # Plot the validation accuracy
    plt.figure()
    plt.plot(range(1, len(accuracy_list) + 1), accuracy_list, "b-")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy Over Epochs")
    plt.grid(True)
    plt.show()

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
def evaluate_model(model, data_loader, device):
    """评估模型的性能，并打印混淆矩阵、准确率、ROC曲线和AUC。

    Args:
        model (nn.Module): 需要评估的模型
        data_loader (DataLoader): 用于评估的数据加载器
        device (torch.device): 训练设备 (e.g., 'cuda' or 'cpu')
    """
    print('Start test!')
    start_time = time.time()

    model.eval()  # 将模型设置为评估状态
    predictions = []
    true_labels = []
    all_outputs = []

    with torch.no_grad():
        for images, labels in data_loader:
            inputs, labels = images.to(device), labels.to(device)
            outputs = model(inputs)
            all_outputs.extend(outputs.tolist())  # 保存所有输出用于ROC曲线

            predictions.extend(torch.argmax(outputs, dim=1).tolist())
            true_labels.extend(labels.tolist())

    print('Test Accuracy of the model on the test images: {} %'.format(
        100 * (torch.argmax(torch.tensor(all_outputs), dim=1) == torch.tensor(true_labels)).sum().item() / len(
            true_labels)))
    end_time = time.time()  # 记录结束时间
    print('Total evaluation time: {:.2f} seconds'.format(end_time - start_time))  # 打印总时间

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(true_labels, predictions)

    # 计算每类的准确率
    #class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    #class_labels = ['1', '2', '3', '4A', '4B', '4C', '5']

    # 计算总准确率
    total_accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)

    # plt.figure(figsize=(5, 4))
    # # 将准确率转换为百分比形式
    # class_accuracies_percent = [x * 100 for x in class_accuracies]
    # total_accuracy_percent = total_accuracy * 100
    # bars = plt.bar(class_labels, class_accuracies_percent, color='#0072B2', alpha=0.7, width=0.7)
    #
    # # 在每个柱子顶部添加准确率文本
    # for bar in bars:
    #     yval = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width() / 2, yval, '{:.1f}'.format(yval), ha='center', va='bottom')
    #
    # # 用虚线画上模型的总准确率
    # plt.axhline(y=total_accuracy_percent, color='r', linestyle='--',
    #             label=f'Total Accuracy: {total_accuracy_percent:.1f}%')
    #
    # plt.ylabel('Accuracy (%)', fontweight='bold')
    # plt.ylim(0, max(class_accuracies_percent) * 1.2)  # 设置纵坐标的上限为最高准确率的120%
    # plt.legend()
    # plt.show()
    #
    # # 打印所有批次的真实标签和所有的预测标签
    # print("True labels:", true_labels)
    # print("Predicted labels:", predictions)
    #
    # # 打印混淆矩阵
    # print("Confusion Matrix:")
    # print(conf_matrix)
    #
    # # 定义自定义的类别标签
    # class_labels = ['1', '2', '3', '4A', '4B', '4C', '5']
    #
    # # 使用Seaborn绘制混淆矩阵
    # plt.figure(figsize=(5, 4))
    # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
    #             xticklabels=class_labels,
    #             yticklabels=class_labels)
    # plt.xlabel('Predicted Label',fontweight='bold')
    # plt.ylabel('True Label',fontweight='bold')
    # plt.show()
    #
    # print('Test Accuracy of the model on the test images: {} %'.format(
    #     100 * (torch.argmax(torch.tensor(all_outputs), dim=1) == torch.tensor(true_labels)).sum().item() / len(
    #         true_labels)))
    #
    # # 计算ROC曲线和AUC
    # num_classes = outputs.shape[1]
    # true_labels = np.array(true_labels)
    # all_outputs = np.array(all_outputs)
    #
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # for i in range(num_classes):
    #     fpr[i], tpr[i], _ = roc_curve(true_labels == i, all_outputs[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    #
    # # 绘制所有类别的ROC曲线
    # plt.figure(figsize=(5,4))
    # #colors = cycle(['red', 'darkorange', 'cornflowerblue','limegreen','darkgoldenrod','deeppink','purple'])
    # # 定义颜色映射
    # colors = ['#4477AA', '#228833', '#EE6677', '#BBBBBB', '#AA3377', '#66CCEE', '#CCBB44']
    # class_labels = [1,2,3]+['4a','4b','4c',5]
    # for i, color in zip(range(num_classes), colors):
    #     class_label = class_labels[i]
    #     plt.plot(fpr[i], tpr[i], color=color, lw=2,
    #              label=f'ROC curve of class {class_label} (AUC = {roc_auc[i]:0.2f})')
    #
    # plt.plot([0, 1], [0, 1], 'k--', lw=2)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('1 - Specificity',fontweight='bold')
    # plt.ylabel('Sensitivity',fontweight='bold')
    # plt.legend(loc="lower right",fontsize=10)
    # plt.show()
    #
    #
    #
    # # 计算灵敏度和特异性
    # sensitivity = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    # # 计算精确度
    # precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
    #
    # # 计算特异性
    # specificity = []
    # for i in range(conf_matrix.shape[0]):
    #     true_negatives = np.sum(conf_matrix) - np.sum(conf_matrix[:, i])-np.sum(conf_matrix[i, :])+conf_matrix[i, i]
    #     false_positives = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]
    #     denominator = true_negatives + false_positives
    #     specificity_i = true_negatives / denominator if denominator != 0 else 0
    #     specificity.append(specificity_i)
    #
    # specificity = np.array(specificity)
    #
    #
    #
    # # 创建一个表格来显示这些指标
    # table_data = np.column_stack(
    #     (sensitivity, precision, specificity * np.ones_like(sensitivity), np.ones_like(sensitivity) * total_accuracy))
    #
    # # 设置类别标签
    # class_labels = ['1', '2', '3', '4A', '4B', '4C', '5']
    #
    # # 绘制表格
    # plt.figure(figsize=(10, 8))
    # ax = plt.subplot()
    # ax.axis('tight')
    # ax.axis('off')
    # the_table = plt.table(cellText=table_data, colLabels=['Sensitivity', 'Precision', 'Specificity', 'Accuracy'],
    #                       rowLabels=class_labels, loc='center', cellLoc='center')
    # the_table.auto_set_font_size(False)
    # the_table.set_fontsize(14)
    # the_table.scale(1.5, 1.5)
    #
    # plt.show()

class MetaLearner(nn.Module):
    def __init__(self, model1, model2, model3):
        super(MetaLearner, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3

        # 定义更深的全连接网络
        self.fc1 = nn.Linear(7 * 3, 128)  # 输入维度：7*3=21，输出：128维度
        self.fc2 = nn.Linear(128, 64)     # 第二层：128 -> 64
        self.fc3 = nn.Linear(64, 7)       # 最终输出层：64 -> 7

        self.relu = nn.ReLU()

    def forward(self, x):
        # 获取每个模型的输出
        with torch.no_grad():  # 假设不再训练基础模型
            out1 = F.softmax(self.model1(x), dim=1)
            out2 = F.softmax(self.model2(x), dim=1)
            out3 = F.softmax(self.model3(x), dim=1)

        # 拼接三个模型的输出
        combined_output = torch.cat((out1, out2, out3), dim=1)

        # 通过更深的网络生成最终分类
        x = self.relu(self.fc1(combined_output))
        x = self.relu(self.fc2(x))
        out = self.fc3(x)

        return out