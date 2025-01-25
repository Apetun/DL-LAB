#!/usr/bin/env python
# coding: utf-8

# ## Week 3 Lab Exercise

# In[67]:


import torch
from matplotlib import pyplot as plt
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.device_count() , device


# ## Lab Exercise

# ### 1) For the following training data, build a linear regression model. Assume w and b are initialized with 1 and learning parameter is set to 0.001.

# x = torch.tensor( [12.4, 14.3, 14.5, 14.9, 16.1, 16.9, 16.5, 15.4, 17.0, 17.9, 18.8, 20.3, 22.4,
# 19.4, 15.5, 16.7, 17.3, 18.4, 19.2, 17.4, 19.5, 19.7, 21.2])
# 
# 
# y = torch.tensor( [11.2, 12.5, 12.7, 13.1, 14.1, 14.8, 14.4, 13.4, 14.9, 15.6, 16.4, 17.7, 19.6,
# 16.9, 14.0, 14.6, 15.1, 16.1, 16.8, 15.2, 17.0, 17.2, 18.6])
# 
# 
# Assume learning rate =0.001. Plot the graph of epoch in x axis and loss in y axis.

# In[68]:


x=torch.tensor([12.4, 14.3, 14.5, 14.9, 16.1, 16.9, 16.5, 15.4, 17.0, 17.9, 18.8, 20.3, 22.4,
19.4, 15.5, 16.7, 17.3, 18.4, 19.2, 17.4, 19.5, 19.7, 21.2])
y=torch.tensor([11.2, 12.5, 12.7, 13.1, 14.1, 14.8, 14.4, 13.4, 14.9, 15.6, 16.4, 17.7, 19.6,
16.9, 14.0, 14.6, 15.1, 16.1, 16.8, 15.2, 17.0, 17.2, 18.6])


# In[69]:


def mse(y, y_pred):
    return torch.mean((y - y_pred)**2)
    
lr = 0.001
epochs = 20
w, b = torch.tensor(1., requires_grad=True), torch.tensor(1., requires_grad=True)
losses = []

for i in range(epochs):
    y_pred = w * x + b

    loss = mse(y, y_pred)

    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        losses.append(loss.detach())
        print(loss)
    w.grad.zero_()
    b.grad.zero_()
        
plt.plot(list(range(1,epochs+1)), losses)
plt.xlabel('No. of Epochs')
plt.ylabel('Loss')
plt.show()


# ### 2) Find the value of w.grad, b.grad using analytical solution for the given linear regression problem. Initial value of w = b =1. Learning parameter is set to 0.001. Implement the same and verify the values of w.grad , b.grad and updated parameter values for two epochs. Consider the difference between predicted and target values of y is defined as (yp-y).

# In[70]:


lr = 0.001
epochs = 2
w, b = torch.tensor(1., requires_grad=True), torch.tensor(1., requires_grad=True)

x = torch.tensor([2, 4])
y = torch.tensor([20, 40])

for i in range(epochs):
    print(w, b)
    y_pred = w * x + b

    loss = mse(y, y_pred)

    loss.backward()
    
    print(f"Epoch {i+1}: w_grad={w.grad.item()}, b_grad={b.grad.item()}")

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    w.grad.zero_()
    b.grad.zero_()
    
    print(f"Epoch {i+1}: Loss={loss.item()}\n")

w, b


# ### 3) Revise the linear regression model by defining a user defined class titled RegressionModel with two parameters w and b as its member variables. Define a constructor to initialize w and b with value 1. Define four member functions namely forward(x) to implement wx+b, update() to update w and b values, reset_grad() to reset parameters to zero, criterion(y, yp) to implement MSE Loss given the predicted y value yp and the target label y. Define an object of this class named model and invoke all the methods. Plot the graph of epoch vs loss by varying epoch to 100 iterations.

# x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0])
# 
# y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0])
# 
# learning_rate = torch.tensor(0.001)

# In[71]:


class RegressionModel:
    def __init__(self, w, b, lr=0.001):
        self.w = torch.tensor(w, requires_grad=True)
        self.b = torch.tensor(b, requires_grad=True)
        self.lr = lr
        self.y = None
        self.loss = None
        self.losses = []
    
    def forward(self, x):
        self.y = self.w * x + self.b
        return self.y
    
    def update(self):
        self.loss.backward()
        with torch.no_grad():
            self.w -= self.lr * self.w.grad
            self.b -= self.lr * self.b.grad
    
    def reset_grad(self):
        self.w.grad.zero_()
        self.b.grad.zero_()
    
    def criterion(self, y, yp):
        self.loss = torch.mean((y - y_pred)**2) 
        self.losses.append(self.loss.detach())
        return self.loss


# In[72]:


model = RegressionModel(1., 1.)
epochs = 100

x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0])
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0])

for i in range(epochs):
    y_pred = model.forward(x)
    loss = model.criterion(y, y_pred)
    model.update()
    model.reset_grad()

plt.plot(np.arange(1, epochs+1), model.losses)
plt.xlabel('No. of Epochs')
plt.ylabel('Loss')
plt.show()


# ### 4) Convert your program written in Qn 3 to extend nn.module in your model. Also override the necessary methods to fit the regression line. Illustrate the use of Dataset and DataLoader from torch.utils.data in your implementation. Use the SGD Optimizer torch.optim.SGD()

# In[73]:


def criterion(yp, y):
    return torch.mean((y - y_pred)**2) 

x = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0])
y = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0])

w = 1.
b = 1.

# Define the RegressionModel class inheriting from nn.Module
class RegressionModel(nn.Module):
    def __init__(self, w, b):
        super(RegressionModel, self).__init__()
        self.w = nn.Parameter(torch.tensor(w, requires_grad=True, dtype=torch.float))
        self.b = nn.Parameter(torch.tensor(b, requires_grad=True, dtype=torch.float))
    
    def forward(self, x):
        return self.w * x + self.b

# Custom Dataset to handle input and output data
class LinearDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

# Create the dataset and data loader
dataset = LinearDataset(x, y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# Initialize the model, loss function, and optimizer
model = RegressionModel(w, b)
#criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        y_pred = model(batch_x)
        
        # Compute loss
        loss = criterion(y_pred, batch_y)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# ### 5) Use PyTorch’s nn.Linear() in your implementation to perform linear regression for the data provided in Qn. 1. Also plot the graph.

# In[74]:


x = torch.tensor( [12.4, 14.3, 14.5, 14.9, 16.1, 16.9, 16.5, 15.4, 17.0, 17.9, 18.8, 20.3, 22.4,
19.4, 15.5, 16.7, 17.3, 18.4, 19.2, 17.4, 19.5, 19.7, 21.2])

y = torch.tensor( [11.2, 12.5, 12.7, 13.1, 14.1, 14.8, 14.4, 13.4, 14.9, 15.6, 16.4, 17.7, 19.6,
16.9, 14.0, 14.6, 15.1, 16.1, 16.8, 15.2, 17.0, 17.2, 18.6])

losses = []

# Create the dataset and data loader
dataset = LinearDataset(x, y)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Initialize the model, loss function, and optimizer
model = nn.Linear(1,1)
#criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        y_pred = model(batch_x)
        
        # Compute loss
        loss = criterion(y_pred, batch_y)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

plt.plot(np.arange(1, num_epochs + 1), losses)
plt.xlabel('No. of Epochs')
plt.ylabel('Loss')
plt.show()


# ### 6) Implement multiple linear regression for the data provided below

# | Subject | X1 | X2 | Y    |
# |---------|----|----|------|
# | 1       | 3  | 8  | -3.7 |
# | 2       | 4  | 5  | 3.5  |
# | 3       | 5  | 7  | 2.5  |
# | 4       | 6  | 3  | 11.5 |
# | 5       | 2  | 1  | 5.7  |
# 
# Verify your answer for the data point X1=3, X2=2.

# In[75]:


x1 = torch.tensor([3, 4, 5, 6, 2])
x2 = torch.tensor([8, 5, 7, 3, 1])
x = torch.stack((x1, x2), dim=1)
y = torch.tensor([-3.7, 3.5, 2.5, 11.5, 5.7])

losses = []

epochs = 20
w, b = [1., 1.], 1.

# Create the dataset and data loader
dataset = LinearDataset(x, y)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Initialize the model, loss function, and optimizer
model = RegressionModel(w, b)
#criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        y_pred = model(batch_x)
        
        # Compute loss
        loss = criterion(y_pred, batch_y)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
    
    losses.append(loss.detach())
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

plt.plot(np.arange(1, num_epochs+1), losses)
plt.xlabel('No. of Epochs')
plt.ylabel('Loss')
plt.show()


# ### 7) Implement logistic regression
# 
# x = [1, 5, 10, 10, 25, 50, 70, 75, 100,]
# 
# y = [0, 0, 0, 0, 0, 1, 1, 1, 1]

# In[76]:


class LogisticModel(nn.Module):
    def __init__(self, w, b):
        super(LogisticModel, self).__init__()
        self.w = nn.Parameter(torch.tensor(w, requires_grad=True, dtype=torch.float))
        self.b = nn.Parameter(torch.tensor(b, requires_grad=True, dtype=torch.float))
    
    def forward(self, x):
        z = self.w * x + self.b
        #print("stuff:", self.w, x, self.b)
        return torch.sigmoid(z)   # 1 / (1 + torch.exp(-z))

x = torch.tensor([1, 5, 10, 10, 25, 50, 70, 75, 100], dtype=torch.float)
y = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float)

losses = []

epochs = 20
w, b = 1., 1.

# Create the dataset and data loader
dataset = LinearDataset(x, y)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Initialize the model, loss function, and optimizer
model = LogisticModel(w, b)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.005)

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    epoch_loss = []
    for batch_x, batch_y in dataloader:
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        y_pred = model(batch_x)
        
        # Compute loss
        loss = criterion(y_pred, batch_y)
        
        # Backward pass
        loss.backward()
        epoch_loss.append(loss.detach())
        
        # Update weights
        optimizer.step()
    
    losses.append(sum(epoch_loss) / len(epoch_loss))
    if (epoch+1) % 10 == 0:
        print(f"Loss for Epoch {epoch+1}/{num_epochs}: {losses[-1]}")

plt.plot(np.arange(1, num_epochs+1), losses)
plt.xlabel('No. of Epochs')
plt.ylabel('Loss')
plt.show()

