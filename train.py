import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import onnxruntime
import onnx
from onnxsim import simplify

# 读取CSV文件
train_csv_path = 'mouse_data.csv'  # 替换为你的CSV文件路径
train_csv = pd.read_csv(train_csv_path, header=None)
test_csv_path = 'mouse_data_test.csv'  # 替换为你的CSV文件路径
test_csv = pd.read_csv(test_csv_path, header=None)

# 提取dx, dy和标签
dx_dy_train = train_csv.iloc[:, 0].apply(lambda x: list(map(int, x.split(','))))
dx_dy_labels_train = train_csv.apply(lambda row: [list(map(int, row[i].split(','))) for i in range(1,11)], axis=1)
dx_dy_test = test_csv.iloc[:, 0].apply(lambda x: list(map(int, x.split(','))))
dx_dy_labels_test = test_csv.apply(lambda row: [list(map(int, row[i].split(','))) for i in range(1,11)], axis=1)

# 转换为PyTorch Tensor
dx_dy_train_tensor = torch.Tensor(dx_dy_train.tolist())
dx_dy_train_tensor = dx_dy_train_tensor.unsqueeze(1)
labels_train_tensor = torch.Tensor(dx_dy_labels_train.tolist())
dx_dy_test_tensor = torch.Tensor(dx_dy_test.tolist())
dx_dy_test_tensor = dx_dy_test_tensor.unsqueeze(1)
labels_test_tensor = torch.Tensor(dx_dy_labels_test.tolist())

# 创建自定义Dataset
class CustomDataset(Dataset):
    def __init__(self, dx_dy, labels):
        self.dx_dy = dx_dy
        self.labels = labels

    def __len__(self):
        return len(self.dx_dy)

    def __getitem__(self, idx):
        return self.dx_dy[idx], self.labels[idx]

# 创建Dataset和DataLoader
train_dataset = CustomDataset(dx_dy_train_tensor, labels_train_tensor)
test_dataset = CustomDataset(dx_dy_test_tensor, labels_test_tensor)
batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# 定义神经网络模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 20)

    def forward(self, input_data):
        x = torch.flatten(input_data, start_dim=1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        return x.view(-1, 10, 2)

# 初始化模型、损失函数和优化器
model = SimpleNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

# 训练模型
epochs = 1000
for epoch in range(epochs):
    for batch_dx_dy, batch_labels in train_dataloader:
        optimizer.zero_grad()
        output = model(batch_dx_dy)
        loss = criterion(output, batch_labels)
        print(loss)
        loss.backward()
        optimizer.step()
        scheduler.step()
print('test')
for idx, data in enumerate(test_dataloader):
        batch_dx_dy, batch_labels = data
        output = model(batch_dx_dy)
        loss = criterion(output, batch_labels)
        print(loss)
        if idx == len(test_dataloader)-1:
            print(batch_dx_dy[0])
            print(output[0])

model.eval()
onnx_name = 'mouse.onnx'
dummy = torch.randn(1, 1, 2)
torch.onnx.export(model, dummy, onnx_name,verbose=True, input_names=['input'], output_names=['output'])