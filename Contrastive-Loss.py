import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
import torch.nn.functional as F
import numpy as np
import os
import random
import sys


if sys.argv[1]== '--load':
    weightname=sys.argv[2]
    tempna='./'
    name=tempna+weightname
    test_only=1
else:
    weightname=sys.argv[2]
    tempna='./'
    name=tempna+weightname
    test_only=0
N=16
class custom_dset(Dataset):
    def __init__(self,
                 img_path,
                 txt_path,
                 img_transform1,
                 img_transform2,
                 ):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            self.img1_list = [
                os.path.join(img_path, i.split()[0]) for i in lines
            ]
            self.img2_list = [
                os.path.join(img_path, i.split()[1]) for i in lines
            ]            
            self.label_list = [i.split()[2] for i in lines]
        self.img_transform1 = img_transform1
        self.img_transform2 = img_transform2
    def __getitem__(self, index):
        img1_path = self.img1_list[index]
        img2_path = self.img2_list[index]
        label = self.label_list[index]
        label=int(label)
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        img1 = img1.astype(np.float)/255
        img2 = img2.astype(np.float)/255
        img1 = cv2.resize(img1,(128,128), interpolation = cv2.INTER_AREA)
        img2 = cv2.resize(img2,(128,128), interpolation = cv2.INTER_AREA)
        img1 = self.img_transform1(img1)
        img2 = self.img_transform2(img2)
        return img1,img2,label
    def __len__(self):
        return len(self.label_list)

class Rescale(object):
    def __call__(self, img):
        if random.random()<0.7:
            f = round(0.1*random.randint(7, 13),2)
            if f>1:
                img = cv2.resize(img,None,fx=f, fy=f, interpolation = cv2.INTER_CUBIC)
                a = int(round((f*128-128)/2))
                img = img[a:a+128,a:a+128]
            else:
                img = cv2.resize(img,None,fx=f, fy=f, interpolation = cv2.INTER_AREA)
                a= int(round((128-f*128)/2))
                temp=np.zeros([128,128,3],dtype=np.uint8)
                temp.fill(0) 
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        temp[i+a,j+a]=img[i,j]
                img=temp
        return img

class Flip(object):
    def __call__(self,img):
        if random.random()<0.7:
            return cv2.flip(img,1)
        return img
        
class Rotate(object):
    def __call__(self,img):
        if random.random()<0.7:
            angle=random.random()*60-30
            rows,cols,cn = img.shape
            M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
            img = cv2.warpAffine(img,M,(cols,rows))
            return img
        return img

class Translate(object):
    def __call__(self,img):
        if random.random()<0.7:
            x=random.random()*20-10
            y=random.random()*20-10
            rows,cols,cn = img.shape
            M= np.float32([[1,0,x],[0,1,y]])
            img = cv2.warpAffine(img,M,(cols,rows))
        return img
            
transform1 = transforms.Compose([Rescale(),Flip(),Translate(),Rotate(),transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 
transform2 = transforms.Compose([Rescale(),Flip(),Translate(),Rotate(),transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])                
train_set = custom_dset('./lfw', './train.txt',transform1,transform2)
train_loader = DataLoader(train_set, batch_size=N, shuffle=True, num_workers=2)

lr = 1e-6
num_epoches = 100

class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            )
        self.conv4 =nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            )
        self.fc = nn.Sequential(
            nn.Linear(131072, 1024),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
        )
  
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
net=Cnn()         
if torch.cuda.is_available() :
    net = net.cuda()  
    
optimizer = torch.optim.Adam(net.parameters(), lr)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((label) * torch.pow(euclidean_distance, 2) + (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive

loss_func = ContrastiveLoss() 
l_his=[]

if test_only==0:
    for epoch in range(num_epoches):
        print('Epoch:', epoch + 1, 'Training...')
        running_loss = 0.0 
        for i,data in enumerate(train_loader, 0):
            image1s,image2s,labels=data
            if torch.cuda.is_available():
                image1s = image1s.cuda()
                image2s = image2s.cuda()
                labels = labels.cuda()
            image1s, image2s, labels = Variable(image1s), Variable(image2s), Variable(labels.float())
            optimizer.zero_grad()
            f1=net(image1s)
            f2=net(image2s)
            loss = loss_func(f1,f2,labels)
            loss.backward()
            optimizer.step()
            if i % 5 == 4:
                l_his.append(loss.data[0])
            # print statistics
            running_loss += loss.data[0]
            if i % 100 == 99:    
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
    print('Finished Training')
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(l_his)    
    plt.xlabel('Steps')  
    plt.ylabel('Loss')  
    fig.savefig('plott2.png')  
    torch.save(net.state_dict(), name)
else:   
    net.load_state_dict(torch.load(name))
    transform = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 
    test_set = custom_dset('./lfw', './train.txt',transform,transform)
    test_loader = DataLoader(test_set, batch_size=N, shuffle=True, num_workers=2)   
    correct = 0
    total = 0
    for data in test_loader:
        image1s,image2s,labels = data
        if torch.cuda.is_available():
            image1s = image1s.cuda()
            image2s = image2s.cuda()
            labels = labels.cuda()
        image1s, image2s, labels = Variable(image1s), Variable(image2s), Variable(labels.float())   
        f1=net(image1s)
        f2=net(image2s)
        dist = F.pairwise_distance(f1, f2)
        dist = dist.cpu()
        for j in range(dist.size()[0]):
            if ((dist.data.numpy()[j]<0.8)):
                if labels.cpu().data.numpy()[j]==1:
                    correct +=1
                    total+=1
                else:
                    total+=1
            else:
                if labels.cpu().data.numpy()[j]==0:
                    correct+=1
                    total+=1
                else:
                    total+=1                
    print('Accuracy of the network on the train images: %d %%' % (
        100 * correct / total))
    
    test_set = custom_dset('./lfw', './test.txt',transform,transform)
    test_loader = DataLoader(test_set, batch_size=N, shuffle=True, num_workers=2)  
    correct = 0
    total = 0
    for data in test_loader:
        image1s,image2s,labels = data
        if torch.cuda.is_available():
            image1s = image1s.cuda()
            image2s = image2s.cuda()
            labels = labels.cuda()
        image1s, image2s, labels = Variable(image1s), Variable(image2s), Variable(labels.float())   
        f1=net(image1s)
        f2=net(image2s)
        dist = F.pairwise_distance(f1, f2)
        dist = dist.cpu()
        for j in range(dist.size()[0]):
            if ((dist.data.numpy()[j]<0.8)):
                if labels.cpu().data.numpy()[j]==1:
                    correct +=1
                    total+=1
                else:
                    total+=1
            else:
                if labels.cpu().data.numpy()[j]==0:
                    correct+=1
                    total+=1
                else:
                    total+=1                
    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))