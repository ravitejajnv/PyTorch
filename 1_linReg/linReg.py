
# coding: utf-8

# In[91]:


x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1,1)
x_train.shape
y_values =[2*i+1 for i in x_values]
y_train = np.array(y_values,dtype=np.float32)
y_train = y_train.reshape(-1,1)


# In[92]:


import torch
import torch.nn as nn
from torch.autograd import Variable


# **Create a Model**
# 
# 1.Linear Model
# 
# -  True Equation: y=2x+1
# 
# 2.Forward
# -  Example
#     -  Input $(x)=1$
#     -  Output $\hat{y}=?$

# In[93]:


#create class

class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        out = self.linear(x)
        return out


# In[94]:


#Instanstiate Model Class

input_dim =1
output_dim=1

model=LinearRegressionModel(input_dim, output_dim)


# In[95]:


#instantiate the loss class

criterion = nn.MSELoss()


# In[96]:


learning_rate=0.01

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# In[106]:


#Train Model
epochs=200

for epoch in range(epochs):
    epoch += 1
    
    #convert numpy arrays to torch Variable
    inputs = Variable(torch.from_numpy(x_train))
    labels = Variable(torch.from_numpy(y_train))
    
    #clear gradients wrt parameters after each epoch
    optimizer.zero_grad()
    
    #Forward to get outputs
    outputs = model(inputs)
    
    #Calculate loss
    loss = criterion(outputs, labels)
    
    #Getting gradients wrt parameters
    loss.backward()
    
    #updating parameters
    optimizer.step()
    
    print('epoch {}, loss {}' .format(epoch, loss.data[0]))


# In[107]:



predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
predicted


# In[108]:


y_train


# In[109]:


#plot the graph
import matplotlib.pyplot as plt

#clear fig
plt.clf()

#Get Predictions
predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()

#plot True data
plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)

#plot predictions
plt.plot(x_train, predicted, '*-', c='red', label='Predictions', alpha=0.8)

plt.legend(loc='best')
plt.show()


# **Save model**

# In[112]:


save_model = False
if save_model is True:
    #saves only parameters
    torch.save(model.state_dict(), 'awesome_model.pkl')


# **Load Model**

# In[113]:


load_model = False
if load_model is True:
    model.load_state_dict(torch.load('awesome_model.pkl'))

