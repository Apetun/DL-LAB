#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install torchvision')


# In[2]:


import torch
from torch.nn import functional as F
import torch.nn as nn


# In[3]:


image = torch.rand(1,1,6,6)
kernal = torch.rand(3,1,3,3)

layer = nn.Conv2d(in_channels=1,out_channels=3,kernel_size=3,padding=0,bias=False,stride=1)
layer.weight = nn.Parameter(kernal)
out1 = layer(image)
out2 = F.conv2d(image,kernal,bias=None,stride=1,padding=0)

print(out1)
print(out2)


# In[4]:


class MNIST(nn.Module):
  def __init__(self):
    super().__init__()
    self.cnn = nn.Sequential(
        nn.Conv2d(1,64,3),
        nn.ReLU(),
        nn.MaxPool2d((2,2),stride=2),
        nn.Conv2d(64,128,3),
        nn.ReLU(),
        nn.MaxPool2d((2,2),stride=2),
        nn.Conv2d(128,64,3),
        nn.ReLU(),
        nn.MaxPool2d((2,2),stride=2)
    )
    self.linear = nn.Sequential(
        nn.Linear(64,20,bias=True),
        nn.ReLU(),
        nn.Linear(20,10,bias=True)
    )
  def forward(self,x):
      features = self.cnn(x)
      batch_size = features.shape[0]
      return self.linear(features.view(batch_size,-1))


# In[5]:


from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transform
import torchvision


# In[6]:


batch_size = 32


# In[7]:


trainset = torchvision.datasets.MNIST(root= "./data/",download=True,train=True,transform=transform.ToTensor())
trainloader = DataLoader(trainset,batch_size=batch_size,shuffle=True)


# In[8]:


testset = torchvision.datasets.MNIST(root= "./data/",download=True,train=False,transform=transform.ToTensor())
testloader = DataLoader(testset,batch_size=batch_size,shuffle=False)


# In[9]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# In[10]:


epochs = 5
model = MNIST().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)


# In[13]:


def one_epoch(index):
  total_loss = 0
  for i,(input,labels) in enumerate(trainloader):
    input, labels = input.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(input)
    loss = criterion(outputs,labels)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
  return total_loss/len(trainloader)*batch_size


# In[14]:


loss_list=[]
for epoch in range(epochs):
  print(epoch)
  epoch_loss = 0
  model.train(True)
  avg_loss=one_epoch(epoch)
  print(avg_loss)
  loss_list.append(avg_loss)

import matplotlib.pyplot as plt
plt.plot(loss_list)
plt.show()


# In[ ]:


model.eval()


# In[19]:


from sklearn.metrics import confusion_matrix
allpred = []
alllabel = []
correct = 0
total = 0
for i,(input,labels) in enumerate(testloader):
    input, labels = input.to(device), labels.to(device)
    output = model(input)
    _,predicted = torch.max(output,dim=1)
    allpred.extend(predicted.cpu().numpy())
    alllabel.extend(labels.cpu().numpy())
    total+=labels.size(0)
    correct+=(predicted == labels).sum()
accuracy = 100*correct/total
print(accuracy)
conf_mat = confusion_matrix(alllabel,allpred)
print(conf_mat)


# In[17]:


total_params = 0
for name,param in model.named_parameters():
    params = param.numel()
    total_params += params
print(total_params)


# In[ ]:





# In[ ]:




