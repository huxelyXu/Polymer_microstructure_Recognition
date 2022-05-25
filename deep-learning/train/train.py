import os
import torch
import torch.nn as nn
import torch.optim as optim
import dataload
from torch.utils.data import DataLoader
from model import vgg16, resnet
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

def set_seed(seed=1):
    #random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


txtpath = r'E:\dataset\pic\result\train.txt'
transform = transforms.Compose([
            #transforms.ToPILImage(),
            #transforms.RandomRotation(90, expand=True),
            #transforms.Grayscale(num_output_channels=3),
            #transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
data = dataload.MyDataset(txtpath, transform=transform)
set_seed()
train_size = int(len(data) * 0.7)
test_size = len(data) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
trainloader = DataLoader(train_dataset, batch_size=30, shuffle=True, num_workers=0)
net = vgg16.VGG('VGG16').cuda()
net_1 = resnet.resnet34().cuda()
loss_save = []
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001,
                       momentum=0.9
                       )

for epoch in range(200):
    train_loss = 0.0
    for batch_idx, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = net_1(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        #if batch_idx % 20 == 19:
            #print('[%d, %5d] loss : %.3f '% (epoch + 1, batch_idx + 1, train_loss / 100))
        print('[%d, %5d] loss : %.3f ' % (epoch + 1, batch_idx + 1, train_loss / 100))
    loss_save.append(train_loss)
    state = {
          'net': net_1.state_dict(),
          'epoch': epoch + 1,
        }
    if not os.path.isdir('checkpoint'):
      os.mkdir('checkpoint')
    if epoch == 0:
        torch.save(state, './checkpoint/epoch_%d.ckpt' % (epoch + 1))
    else:
        if train_loss <= loss_save[-1]:
            torch.save(state, './checkpoint/epoch_%d.ckpt' % (epoch + 1))
            print('saving epoch %d model ...' % (epoch + 1))
    train_loss = 0.0
print('Finished training')

x = np.arange(len(loss_save))
y = loss_save
plt.plot(x, y, color='r', marker='o', linestyle='--')
plt.show()



