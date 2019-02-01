import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import csv
import pandas as pd
import numpy as np
from numpy import loadtxt
from torch.utils.data.dataset import Dataset
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F
import tensorflow as tf
from torch.utils.data.sampler import SubsetRandomSampler
import skorch
import sklearn



device = torch.device('cuda')
num_epochs = 40
num_classes = 2
batch_size = 1
learning_rate = 0.01

class CustomDatasetFromCSV(Dataset):
    def __init__(self, csv_path,a,b,c,d,length):

        self.transforms = transforms
        #self.transforms = transforms.RandomCrop(64)
        self.to_tensor = transforms.ToTensor()
        self.data_info = pd.read_csv(csv_path, header=None)
        if c == 0:
            self.image_arr = np.asarray(self.data_info.iloc[a:b, 0])
            self.label_arr = np.asarray(self.data_info.iloc[a:b, 1])
        else:
            self.image_arr = np.concatenate((np.asarray(self.data_info.iloc[a:b, 0]),np.asarray(self.data_info.iloc[c:d, 0])))
            self.label_arr = np.concatenate((np.asarray(self.data_info.iloc[a:b, 1]),np.asarray(self.data_info.iloc[c:d, 1])))
        self.image_len = length

    def __getitem__(self, index):

        single_image_name = self.image_arr[index]
        lines_img = loadtxt(single_image_name +'.txt', delimiter=",", unpack=False)
        lines_img = np.array(lines_img)
        lines_img = np.reshape(lines_img, (1,64,10240))

        img_as_tensor = torch.cuda.FloatTensor(lines_img)
        #img_as_tensor = torch.from_numpy(lines_img)


        single_image_label = self.label_arr[index]
        #single_image_label = ord(single_image_label[0])
        #single_image_label = np.array(single_image_label)
        #single_image_label = np.transpose(single_image_label)
        single_image_label = [single_image_label]
        single_image_label = torch.cuda.FloatTensor(single_image_label).long()



        return (img_as_tensor, single_image_label)


    def __len__(self):
        return self.image_len

class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 20, (1,11)),
            nn.Conv2d(20, 20, (64,1)),
            nn.BatchNorm2d(20),
            nn.ReLU(True),
            nn.MaxPool2d((1,3)))
        self.layer2 = nn.Sequential(
            nn.Conv2d(20, 20, (1,11)),
            nn.BatchNorm2d(20),
            nn.ReLU(True),
            nn.MaxPool2d((1,4)))
        self.layer3 = nn.Sequential(
            nn.Conv2d(20, 20, (1,11)),
            nn.BatchNorm2d(20),
            nn.ReLU(True),
            nn.MaxPool2d(1,3))
        self.layer4 = nn.Sequential(
            nn.Conv2d(20, 20, (1,11)),
            nn.BatchNorm2d(20),
            nn.ReLU(True),
            nn.MaxPool2d((1,3)))
        self.drop_out = nn.Dropout(0.2)
        self.fc1 = nn.Linear(20*1*90, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 3)




        #self.fc2 = nn.Linear(100,2)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0),-1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out



#transformations = transforms.Compose([transforms.ToTensor()])





ns = 95
for fold in range (1,11):
    if fold == 1:
        a1_train = (fold+1)*ns+1
        a2_train = 950
        a1_test = (fold-1)*ns+1
        a2_test = fold*ns
        custom_mnist_from_csv_train = CustomDatasetFromCSV('...\\label.csv',a1_train,a2_train,0,0,758)
        custom_mnist_from_csv_test = CustomDatasetFromCSV('...\\label.csv',a1_test,a2_test,0,0,94)
    elif fold == 10:
        a1_train = 1
        a2_train = (fold-2)*ns
        a1_test = (fold-1)*ns+1
        a2_test = 950
        custom_mnist_from_csv_train = CustomDatasetFromCSV('C:\\Users\\Seyed\\PycharmProjects\\MindWandering\\label.csv',a1_train,a2_train,0,0,758)
        custom_mnist_from_csv_test = CustomDatasetFromCSV('C:\\Users\\Seyed\\PycharmProjects\\MindWandering\\label.csv',a1_test,a2_test,0,0,94)
    else:
        a1_train = 1
        a2_train = (fold-1)*ns
        b1_train = (fold+1)*ns+1
        b2_train = 950
        a1_test = (fold-1)*ns+1
        a2_test = fold*ns
        custom_mnist_from_csv_train = CustomDatasetFromCSV('C:\\Users\\Seyed\\PycharmProjects\\MindWandering\\label.csv',a1_train,a2_train,b1_train,b2_train,758)
        custom_mnist_from_csv_test = CustomDatasetFromCSV('C:\\Users\\Seyed\\PycharmProjects\\MindWandering\\label.csv',a1_test,a2_test,0,0,94)



    dataset = torch.utils.data.DataLoader(dataset=custom_mnist_from_csv_train, batch_size=1, shuffle=True)

    model = ConvNet(num_classes).to(device)
    model = model.cuda()


    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(dataset)

    sumtrain = 0
    correct = 0
    loss = 0
    for epoch in range(num_epochs):
        print("train accuracy = ", sumtrain)
        print("loss=", loss)
        sumtrain = 0
        correct = 0
        for i, (images, labels) in enumerate(dataset):
            images = images.to(device)
            labels = labels.to(device)
            #print(labels)



            outputs = model(images)


            #output = Variable(outputs.float())
            #labels = Variable(torch.FloatTensor(10).uniform_(0, 120).long())

            labels = labels.view(-1)
            loss = criterion(outputs, labels)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            outputs = torch.cuda.FloatTensor(outputs).long()

            correct = 0


            if (outputs[0,0]<outputs[0,1]) and torch.eq(labels,1):
                correct = correct+1
            elif (outputs[0,0]>outputs[0,1]) and torch.eq(labels,0):
                correct = correct+1

            sumtrain = correct+sumtrain

    sumtest = 0
    correct = 0

    dataset = torch.utils.data.DataLoader(dataset=custom_mnist_from_csv_test, batch_size=1, shuffle=True)



    for i, (images, labels) in enumerate(dataset):
        images = images.to(device)
        labels = labels.to(device)


        outputs = model(images)


        outputs = torch.cuda.FloatTensor(outputs).long()

        correct = 0


        if (outputs[0,0]<outputs[0,1]) and torch.eq(labels,1):
            correct = correct+1
        elif (outputs[0,0]>outputs[0,1]) and torch.eq(labels,0):
            correct = correct+1

        sumtest = correct+sumtest
        print("test accuracy = ", sumtest)



