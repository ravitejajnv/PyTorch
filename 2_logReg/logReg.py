
# coding: utf-8

# In[11]:


import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable


# In[12]:


#Loading Dataset
train_dataset = dsets.MNIST(root='./data',
                            train=True, 
                            transform=transforms.ToTensor(),
                           download=False)
test_dataset = dsets.MNIST(root='./data', 
                           train=False, 
                           transform=transforms.ToTensor())


# In[13]:


batch_size = 100
n_iters = 6000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

# making dataset iterable

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                         batch_size=batch_size, 
                                         shuffle=False)


# In[14]:


#create model class
class LogisticRegressionMoel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegressionMoel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        out = self.linear(x)
        return out

#instanstiate model class
input_dim = 28*28
output_dim = 10

model = LogisticRegressionMoel(input_dim, output_dim)

#instanstiate loss class
criterion = nn.CrossEntropyLoss()

#instanstiate optimizer class
learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# In[15]:


#train the model

iter = 0
for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        images = Variable(images.view(-1,28*28))
        labels = Variable(labels)
        
        #clear gradients wrt parameters
        optimizer.zero_grad()
        
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        #update parameters
        optimizer.step()
        
        iter +=1
        
        if iter%500 == 0:
            #calculate accuracy
            correct = 0
            total = 0
            
            #iterate to test dataset
            for images, labels in test_loader:
                images = Variable(images.view(-1, 28*28))
                
                #forward pass only to get logits/output
                outputs = model(images)
                
                _,predicted = torch.max(outputs.data,1)
                
                total += labels.size(0)
                correct += (predicted==labels).sum()
            
            accuracy = 100 * correct/total
            
            print('Iterations: {}. Loss: {}. Accuracy: {}' .format(iter, loss.data[0], accuracy))

