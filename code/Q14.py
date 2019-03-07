# --------------------------------------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

#torch.manual_seed(1122)
start_time = time.time()
# --------------------------------------------------------------------------------------------
# Choose the right values for x.
input_size = 28*28
hidden_size = 60
num_classes = 10
num_epochs = 20
batch_size = 128
learning_rate = 0.001
# --------------------------------------------------------------------------------------------
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# --------------------------------------------------------------------------------------------
train_set = torchvision.datasets.FashionMNIST(root='./data_fashion', train=True, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.FashionMNIST(root='./data_fashion', train=False, download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# Find the right classes name. Save it as a tuple of size 10.
classes = ("T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot")
# --------------------------------------------------------------------------------------------

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

dataiter = iter(train_loader)
images, labels = dataiter.__next__()

imshow(torchvision.utils.make_grid(images))
plt.show()
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# --------------------------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,num_classes)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)

    def forward(self, x):

        out = self.fc1(x)
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc3(out)

        return out

# --------------------------------------------------------------------------------------------
# Choose the right argument for x
net = Net(input_size, hidden_size, num_classes)
#net = net.cuda()
# --------------------------------------------------------------------------------------------
# Choose the right argument for x
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
# --------------------------------------------------------------------------------------------
step = []
loss_step = []

train_pred_list=[]
train_lab_list=[]

total_train = 0
correct_train = 0

net.train()
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        images, labels = data
        images = images.view(-1,1 * 28 * 28)#.cuda()
        images, labels = Variable(images), Variable(labels)#.cuda())
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_set) // batch_size, loss.item()))
            loss_step.append(loss.item())

fig1 = plt.figure(figsize=(5,5))
fig1 = plt.plot(np.arange(0,len(loss_step*100),100),loss_step,color='red')
plt.title('Loss over steps (' + str(len(loss_step*100)) + ')')
plt.xlabel("steps")
plt.ylabel("error")
plt.show()

# # --------------------------------------------------------------------------------------------
# # There is bug here find it and fix it
correct = 0
total = 0

test_pred_list=[]
test_lab_list=[]

net.eval()
for images, labels in test_loader:
    images = Variable(images.view(-1,1* 28 * 28))#.cuda()
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum().item()
    test_pred = predicted.cpu().numpy()
    test_lab = labels.cpu().numpy()
    test_pred_list.extend(test_pred)
    test_lab_list.extend(test_lab)

cnf_mat = confusion_matrix(test_pred_list,test_lab_list)
print("Confusion Matrix - ")
print(cnf_mat)
print(120*"+")
mod_acc = precision_recall_fscore_support(test_pred_list,test_lab_list,average='macro')
print("Precision, Recall, F1Score & Support of this model is " + str(mod_acc))
print(120*"+")

print('Accuracy of the network on the 10000 test images: %.2f%%' % (100 * correct / total))
print('Misclassification of the network on the 10000 test images: %.2f%%' % (100 - 100 * correct / total))
# --------------------------------------------------------------------------------------------
_, predicted = torch.max(outputs.data, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
# --------------------------------------------------------------------------------------------
# There is bug here find it and fix it
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in test_loader:
    images, labels = data
    images = Variable(images.view(-1,1* 28 * 28))#.cuda()
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    labels = labels.cpu().numpy()
    c = (predicted.cpu().numpy() == labels)
    for i in range(10):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1

print(120*"+")

# ---------------------------------------a-----------------------------------------------------
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
# --------------------------------------------------------------------------------------------
torch.save(net.state_dict(), 'model.pkl')
execution_time = time.time() - start_time
print(120*"+")
print('Execution time is %0.5f minutes'%(execution_time/60))