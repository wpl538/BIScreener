import torch
import cv2
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from utils import CustomDataset, set_random_seed, initialize_model_fc, EarlyStopping, train_with_early_stopping
from torchvision import models
from torch import nn, optim

print(torch.cuda.is_available())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    # 读取CSV文件
    df1 = pd.read_csv(r'E:\pl\WPL\original_image.csv')
    # 分割数据集
    train_df1 = df1[df1['dataset_type'] == 'train']
    val_df1 = df1[df1['dataset_type'] == 'valid']
    test_df1 = df1[df1['dataset_type'] == 'test']

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),  # 以50%的概率随机水平翻转图像
        transforms.RandomVerticalFlip(p=1.0),  # 以50%的概率随机垂直翻转图像
        transforms.Resize((224, 224)),
        transforms.ToTensor()  # 将图像转换为Tensor格式
    ])

    # 预处理之后的彩色图像
    train_dataset1 = CustomDataset(train_df1, transform=transform)
    val_dataset1 = CustomDataset(val_df1, transform=transform)
    test_dataset1 = CustomDataset(test_df1, transform=transform)

    # 数据集的加载
    BS = 32 #batch_size
    train_loader = DataLoader(train_dataset1, batch_size=BS, shuffle=True)
    valid_loader = DataLoader(val_dataset1, batch_size=BS, shuffle=False)
    test_loader = DataLoader(test_dataset1, batch_size=BS, shuffle=False)
    print('Number of training samples:',len(train_dataset1))
    print('Number of training samples:',len(val_dataset1))
    print('Number of training samples:',len(test_dataset1))

    set_random_seed(102) #固定了随机种子，保证结果重现性，模型不好记得修改
    model1 = models.resnet18(pretrained=True)
    model1 = initialize_model_fc(model1, num_classes = 7, learning_rate = 0.001, device = "cuda", last_layer_name='fc')# 初始化模型的最后一个全连接层
    print(model1)

    # model2 = models.vgg16(pretrained=True)
    # num_features = model2.classifier[6].in_features
    # num_classes = 7  # 新分类任务的类别数
    # model2.classifier[6] = nn.Linear(num_features, num_classes).to(DEVICE)
    # model2.classifier[6].weight.requires_grad = True
    # model2.classifier[6].bias.requires_grad = True
    # learning_rate = 0.0005  # 全局学习率
    # new_fc_learning_rate = learning_rate * 10  # 新全连接层的学习率
    # for param in model2.classifier[6].parameters():
    #     param.requires_grad = True
    #     if param.dim() > 1:  # 权重参数
    #         nn.init.xavier_uniform_(param)
    #         param.requires_grad = True
    #         param.lr = new_fc_learning_rate
    #     else:  # 偏置参数
    #         nn.init.constant_(param, 0)
    #         param.requires_grad = True
    #         param.lr = new_fc_learning_rate
    # print(model2)

    # model3 = models.densenet161(pretrained=True)
    # model3= initialize_model_fc(model3, num_classes=7, learning_rate=0.001, device="cuda", last_layer_name='classifier')
    # print(model3)

    # 定义损失函数和优化器
    weights = torch.tensor([0.7846, 1.5157, 0.8661, 1.2350, 1.4820, 0.6805, 0.4359]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.SGD(model1.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)   #修改参数

    #训练
    num_epochs = 1000
    early_stopper = EarlyStopping(patience=50, delta=0, verbose=True, path='.\models\checkpoint_resnet18_102.pt') #最后一个数字是随机种子
    train_with_early_stopping(model1, train_loader, criterion, optimizer, num_epochs, valid_loader, early_stopper)    #修改你的模型

