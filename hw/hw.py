from jittor.dataset.cifar import CIFAR10
import jittor as jt
import pygmtools as pygm
import numpy as np
import random
from jittor import nn, Module
import sys, os
import math 
from jittor import init
if jt.has_cuda:
    jt.flags.use_cuda = 1
import matplotlib.pyplot as plt
import pylab as pl

pygm.BACKEND = 'jittor'

images_train = jt.array(np.load('images_train.npy'))
labels_train = jt.array(np.load('labels_train.npy'))
images_test = jt.array(np.load('images_test.npy'))
labels_test = jt.array(np.load('labels_test.npy'))

batch_size = 1
learning_rate = 0.01
momentum = 0.9
weight_decay = 1e-4
epochs = 50

class Model (Module):
    def __init__ (self):
        super (Model, self).__init__()
        self.conv1 = nn.Conv(3, 32, 3)
        self.conv2 = nn.Conv(32, 64, 3)
        self.bn = nn.BatchNorm(64)
        self.max_pool = nn.Pool (2, 2)
        self.relu = nn.Relu()
        self.fc1 = nn.Linear (64*6*6, 256)
        self.fc2 = nn.Linear (1024, 16)
    
    def getf (self, x):
        x = self.conv1 (x)
        x = self.relu (x)
        x = self.conv2 (x)
        x = self.bn (x)
        x = self.relu (x)
        x = self.max_pool (x)
        x = jt.reshape (x, [x.shape[0], -1])
        x = self.fc1 (x)
        return x
    
    def execute (self, x) :
        x = self.getf(x)
        x = x.reshape(-1)
        x = self.fc2(x)
        x = x.reshape(4,4)
        x = pygm.sinkhorn(x)
        x = x.reshape(-1)
        return x

model = Model()
lossfunc = nn.MSELoss()
optimizer = nn.SGD(model.parameters(), learning_rate, momentum, weight_decay)
model.train()
for epoch in range(epochs):
    model.train()
    for i in range(labels_train.shape[0]):
        outputs = model(images_train[i])
        targets = np.array(labels_train[i])
        loss = lossfunc.execute(outputs, targets)
        optimizer.step (loss)
    print(epoch)
    model.eval()
    total_acc = 0
    total_num = 0
    for i in range(labels_test.shape[0]):
        outputs = model(images_test[i])
        targets = np.array(labels_test[i])
        outputs = outputs.reshape(4,4)
        targets = targets.reshape(4,4)
        pred = np.argmax(outputs.numpy(), axis=1)
        gt = np.argmax(targets, axis=1)
        acc0 = (gt == pred)
        acc = acc0[0]*acc0[1]*acc0[2]*acc0[3]
        total_acc += acc
        total_num += 1
    print(total_acc/total_num)

model_path = './cifar10_model.pkl'
model.save(model_path)


