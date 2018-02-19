
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable


# In[5]:


train_dataset = dsets.MNIST(root='../ffnn/data',
                           train=True,
                           transform=transforms.ToTensor(),
                           download=True)
test_dataset = dsets.MNIST(root='../ffnn/data',
                          train=False,
                          transform=transforms.ToTensor())


# In[6]:


batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(train_dataset)/batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                         batch_size=batch_size,
                                         shuffle=False)


# In[12]:


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(32*7*7, 10)
        
    def forward(self,x):
        
        out = self.cnn1(x)
        out = self.relu1(out)        
        out = self.maxpool1(out)
        
        out = self.cnn2(out)
        out = self.relu2(out)        
        out = self.maxpool2(out)
        
        #Resize, i.e., flatten it: 100 becasue of the batch_size
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        return out


# In[13]:


model = CNNModel()


# In[14]:


criterion = nn.CrossEntropyLoss()


# In[15]:


learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# In[18]:


iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        optimizer.step()
        
        iter += 1
        
        if iter % 500 == 0:
            #calculate accuracy
            correct = 0
            total = 0
            
            for images, labels in test_loader:
                images = Variable(images)
                outputs = model(images)
                
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / total
            
            print('Iterations: {}. Loss: {}. Accuracy: {}' .format(iter, loss.data[0], accuracy))

