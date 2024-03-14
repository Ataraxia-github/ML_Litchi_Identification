import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import time

from Model import Classifier
from data_generation import *

batch_size = 128

''' Dataset 数据处理'''
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),  # 随机将图片水平翻转
    transforms.RandomRotation(15),  # 随机旋转图片 (-15,15)
    transforms.ToTensor(),  # 将图片转成 Tensor, 并把数值normalize到[0,1](data normalization)
])
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])


''' 总数据集 '''
all_x, all_y, test_x = data_generation()
all_set = ImgDataset(all_x, all_y, train_transform)
test_set = ImgDataset(test_x, transform=test_transform)
print("---------------Dataset complicated---------------")


''' 划分训练集及验证集 '''
all_set_size = len(all_set)
validation_split = 0.2
validation_size = int(all_set_size * validation_split)

# 随机划分数据集
train_indices, validation_indices = train_test_split(range(all_set_size), test_size=validation_split, random_state=42)

# 根据划分的索引创建训练集和验证集
train_subset = torch.utils.data.Subset(all_set, train_indices)
validation_subset = torch.utils.data.Subset(all_set, validation_indices)

# 创建 DataLoader 加载数据
train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(validation_subset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
print("---------------DataLoader complicated---------------")


#模型训练
''' Training '''
print("---------------Training---------------")
model = Classifier().cpu()
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # optimizer 使用 Adam
num_epoch = 1  # 迭代

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        train_pred = model(data[0].cpu())
        batch_loss = loss(train_pred, data[1].cpu())
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cpu())
            batch_loss = loss(val_pred, data[1].cpu())

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
              (epoch + 1, num_epoch, time.time() - epoch_start_time,
               train_acc / train_subset.__len__(), train_loss / train_subset.__len__(), val_acc / validation_subset.__len__(),
               val_loss / validation_subset.__len__()))


''' Testing '''
model.eval()
test_dir = 'data/test'
prediction = []
image_dir = list(sorted(os.listdir(test_dir)))

with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model(data.cpu())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)
with open("predict1.csv", 'w') as f:
    f.write('picture,Category\n')
    for t, y in zip(image_dir,prediction):
        f.write('{},{}\n'.format(t, y))
print("Testing complicated")

