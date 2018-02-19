
# coding: utf-8

# In[8]:


import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable


# In[9]:


train_dataset = dsets.MNIST(root='./data', 
                           train=True, 
                           transform=transforms.ToTensor(), 
                           download=True)
test_dataset = dsets.MNIST(root='./data', 
                          train=False, 
                          transform=transforms.ToTensor())


# In[10]:


#make datasets iterable

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


# In[11]:


# Create model class
class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FeedforwardNeuralNetModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        
    def forward(self,x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        
        out = self.fc3(out)
        return out


# In[12]:


#instantiate model class
input_dim = 28*28
hidden_dim = 100
output_dim = 10

model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)


# In[13]:


#instantiate loss class
criterion = nn.CrossEntropyLoss()

#instantiate optimizer class
learning_rate = 0.1

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# In[15]:


iter = 0
for epoch in range (num_epochs+1):
    for i, (images, labels) in enumerate(train_loader):
        #Load images as variables
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        optimizer.step()
        
        iter +=1
        
        if iter % 500 == 0:
            #calculate accuracy
            correct = 0
            total = 0
            #iterate through test dataset
            for images, labels in test_loader:
                images = Variable(images.view(-1,28*28))
                
                outputs = model(images)
                
                _, predicted = torch.max(outputs.data, 1)
                             
                total += labels.size(0)
               
                correct += (predicted == labels).sum()
            
            accuracy = 100 * correct / total
            
            print('Iterations: {}. Loss: {}. Accuracy: {}.' .format(iter, loss.data[0], accuracy))
                

