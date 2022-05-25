import torch
from model import vgg16
from torch.utils.data import DataLoader
import dataload
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
def set_seed(seed=1):
    #random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

net = vgg16.VGG('VGG16')
net = net.cuda()
txtpath = r'E:\dataset\pic\result\train.txt'
data = dataload.MyDataset(txtpath)
set_seed()
train_size = int(len(data) * 0.7)
test_size = len(data) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
testloader = DataLoader(test_dataset, batch_size=60, shuffle=True, num_workers=0)
checkpoint = torch.load('./checkpoint/epoch_400.ckpt')
net.load_state_dict(checkpoint['net'])
start_epoch = checkpoint['epoch']

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images.cuda())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum().item()
print('Accuracy of the network on the  test images: %d %%' % (100*correct/total))

images, labels = data
outputs = net(images.cuda())
_, predicted = torch.max(outputs.data, 1)
y_test = np.array(labels.cpu(), dtype='<U1')
predictions_labels = np.array(predicted.cpu(), dtype='<U1')
labels = [0, 1, 2, 3, 4]
y_true = y_test  # 正确标签
y_pred = predictions_labels  # 预测标签
tick_marks = np.array(range(len(labels))) + 0.5
print(u'预测结果:')
print(predictions_labels)
print(u'算法评价:')
print(classification_report(y_test, predictions_labels))
cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm_normalized)
plt.figure(figsize=(12, 8), dpi=120)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.01:
        plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
# show confusion matrix
plt.savefig('matrix.png', format='png')
plt.show()