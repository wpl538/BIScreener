import torch
import numpy as np
import pandas as pd
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from utils import CustomDataset, set_random_seed, initialize_model_fc, EarlyStopping, train_with_early_stopping, MetaLearner, evaluate_model
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
    BS = 32  # batch_size
    #train_loader = DataLoader(train_dataset1, batch_size=BS, shuffle=True)
    #valid_loader = DataLoader(val_dataset1, batch_size=BS, shuffle=False)
    test_loader = DataLoader(test_dataset1, batch_size=BS, shuffle=False)
    #print('Number of training samples:', len(train_dataset1))
    #print('Number of training samples:', len(val_dataset1))
    print('Number of training samples:', len(test_dataset1))

    # 假设你有三个基础模型，已经训练好
    model1 = models.resnet18(pretrained=False)
    model1 = initialize_model_fc(model1, num_classes=7, learning_rate=0.001, device = DEVICE, last_layer_name='fc')
    model1.load_state_dict(torch.load('.\models\checkpoint_resnet18_106.pt', map_location=DEVICE))
    model1.to(DEVICE)

    model2 = models.vgg16(pretrained=False)
    num_features = model2.classifier[6].in_features
    num_classes = 7  # 新分类任务的类别数
    model2.classifier[6] = nn.Linear(num_features, num_classes).to(DEVICE)
    model2.load_state_dict(torch.load('.\models\checkpoint_vgg16_109.pt', map_location=DEVICE))
    model2.to(DEVICE)

    model3 = models.densenet161(pretrained=False)
    model3 = initialize_model_fc(model3, num_classes=7, learning_rate=0.001, device = DEVICE)
    model3.load_state_dict(torch.load('.\models\checkpoint_densenet161_110.pt', map_location=DEVICE))
    model3.to(DEVICE)


    meta_learner = MetaLearner(model1, model2, model3)
    meta_learner.train()  # 设为训练模式

    # 定义损失函数和优化器
    weights = torch.tensor([0.7846,1.5157,0.8661,1.2350,1.4820,0.6805,0.4359]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(meta_learner.parameters(), lr=0.005)

    #训练
    # num_epochs = 1000
    # early_stopper = EarlyStopping(patience=50, delta=0, verbose=True, path='.\models\checkpoint_MetaLearner3.pt')
    # train_with_early_stopping(meta_learner, train_loader, criterion, optimizer, num_epochs, valid_loader, early_stopper)

    #预测
    meta_learner.load_state_dict(torch.load('.\models\checkpoint_MetaLearner.pt', map_location=DEVICE))
    meta_learner.to(DEVICE)
    start_time = time.time()
    # evaluate_model(meta_learner,train_loader, device=DEVICE)
    # evaluate_model(meta_learner,valid_loader, device=DEVICE)
    evaluate_model(meta_learner,test_loader, device=DEVICE)
    end_time = time.time()

    print(f"预计耗时：{end_time - start_time}秒")
