import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 自定义数据集类
class PoseDataset(Dataset):
    def __init__(self, X, y):
        """
        参数：
            X: 特征数组，形状为 [样本数, 55]
            y: 标签数组，形状为 [样本数]
        """
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # 返回 tensor 格式的特征和标签
        sample = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return sample, label

# 1. 数据预处理
# 假设数据存储在 "pose_data.csv" 中，第一列为 label，其余 55 列为特征
data = pd.read_csv("pose.csv")
y = data.iloc[:, 0].values   # 标签
X = data.iloc[:, 1:].values  # 特征

# 划分训练集和验证集（例如 80% 训练，20% 验证）
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 对特征进行标准化（均值为0，标准差为1），仅基于训练集计算
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# 创建数据集对象和 DataLoader
train_dataset = PoseDataset(X_train, y_train)
val_dataset = PoseDataset(X_val, y_val)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 2. 定义分类器模型
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# 设定输入维度为 55（4+17*3），类别数根据数据集设置
input_dim = X_train.shape[1]  # 应该为 55
num_classes = len(np.unique(y))
model = Classifier(input_dim, num_classes)

# 3. 设置训练参数
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 4. 训练和验证循环
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    train_correct = 0
    total_train = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * X_batch.size(0)
        _, preds = torch.max(outputs, 1)
        total_train += y_batch.size(0)
        train_correct += (preds == y_batch).sum().item()

    avg_train_loss = train_loss / total_train
    train_acc = train_correct / total_train

    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_correct = 0
    total_val = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, 1)
            total_val += y_batch.size(0)
            val_correct += (preds == y_batch).sum().item()
    avg_val_loss = val_loss / total_val
    val_acc = val_correct / total_val

    print(f"Epoch [{epoch+1}/{num_epochs}]: "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

# 训练结束后保存模型
torch.save(model.state_dict(), "pose_classifier_high_acc.pth")
