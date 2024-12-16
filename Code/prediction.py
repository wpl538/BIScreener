import torch
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from utils import CustomDataset, set_random_seed, initialize_model_fc,  evaluate_model
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

    set_random_seed(110)  # 固定了随机种子，保证结果重现性，模型不好记得修改
    # model1 = models.resnet18(pretrained=False)
    # model1 = initialize_model_fc(model1, num_classes=7, learning_rate=0.001, device = DEVICE, last_layer_name='fc')
    # model1.load_state_dict(torch.load('.\models\checkpoint_resnet18_106.pt', map_location=DEVICE))
    # model1.to(DEVICE)
    # evaluate_model(model1, test_loader, device = DEVICE)

    model2 = models.vgg16(pretrained=False)
    num_features = model2.classifier[6].in_features
    num_classes = 7  # 新分类任务的类别数
    model2.classifier[6] = nn.Linear(num_features, num_classes).to(DEVICE)
    model2.load_state_dict(torch.load('.\models\checkpoint_vgg16_109.pt', map_location=DEVICE))
    model2.to(DEVICE)
    evaluate_model(model2, test_loader, device=DEVICE)

    # model3 = models.densenet161(pretrained=False)
    # model3 = initialize_model_fc(model3, num_classes=7, learning_rate=0.001, device=DEVICE)
    # model3.load_state_dict(torch.load('.\models\checkpoint_densenet161_3.pt', map_location=DEVICE))
    # model3.to(DEVICE)
    # evaluate_model(model3, test_loader, device=DEVICE)



