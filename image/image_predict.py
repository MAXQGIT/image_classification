import pickle
import torch.nn as nn
import torch
from PIL import Image
import cv2
from torchvision import transforms, datasets


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


with open('label.pkl', 'rb') as f:
    data = pickle.load(f)
label = {v: k for k, v in data.items()}

model = torch.load('model.pt')
img_path = 'image_classification_data/seg_pred/3.jpg'
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])


def process_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    # img = Image.open(img_path)
    img = transform(img)
    img = img.unsqueeze(0)  # 增加batch维度
    return img


image = process_image(img_path)
image = image.cuda()
outputs = model(image)
_, predicted = torch.max(outputs, 1)
pre = predicted.tolist()[0]
img = cv2.imread(img_path)
cv2.imshow(label[pre],img)
cv2.waitKey()
print(label[pre])
