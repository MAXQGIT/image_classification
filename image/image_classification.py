import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import torch.optim as optim
import pickle
from torch.optim.lr_scheduler import StepLR

device = 'cuda' if torch.cuda.is_available() else 'cpu'
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

train_data_dir = 'image_classification_data/seg_train'
train_dataset = datasets.ImageFolder(root=train_data_dir, transform=transform)

# 获取类标签映射
class_to_idx = train_dataset.class_to_idx
# 数据标签的保存
with open('label.pkl', 'wb') as f:
    pickle.dump(class_to_idx, f)

train_dataloder = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_data_dir = 'image_classification_data/seg_test'
test_dataset = datasets.ImageFolder(root=test_data_dir, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 6, 3, 1)
        self.maxpooling = nn.MaxPool2d(3, 1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(6 * 216 * 216, 128)
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.maxpooling(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.maxpooling(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def val_model(model, test_dataloader, criterion):
    correct, total = 0, 0
    for img, label in test_dataloader:
        img = img.to(device)
        label = label.to(device)
        outputs = model(img)
        test_loss = criterion(outputs, label)
        _, outputs = torch.max(outputs.data, dim=1)
        correct += (outputs == label).sum().item()
        total += label.size(0)
    test_correct = correct / total
    return test_loss, test_correct


model = CNN_Model().to(device)
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

for epoch in range(30):
    correct, total = 0, 0
    for img, label in train_dataloder:
        img = img.to(device)
        label = label.to(device)
        outputs = model(img)
        train_loss = criterion(outputs, label)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        _, outputs = torch.max(outputs.data, dim=1)
        correct += (outputs == label).sum().item()
        total += label.size(0)
    scheduler.step()
    train_correct = correct / total
    model.eval()
    test_loss, test_correct = val_model(model, test_dataloader, criterion)
    torch.save(model, 'model.pt')
print('epoch:{} train_loss:{} train_correct:{:.2f}% test_loss:{} test_correct:{:.2f}%'.format(epoch,
                                                                                                  train_loss.item(),
                                                                                                  100 * train_correct,
                                                                                                  test_loss.item(),
                                                                                                  100 * test_correct))
