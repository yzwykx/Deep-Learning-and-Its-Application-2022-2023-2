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

np.random.seed(0)

train_set = CIFAR10()
test_set = CIFAR10(train=False)
train_set.set_attrs(batch_size=1)
test_set.set_attrs(batch_size=1)
images_train = []
labels_train = []

for imgs, labs in train_set:
    imgs = np.array(imgs)
    timage = []
    list = [1,2,3,4]
    random.shuffle(list)
    tlabel = []
    for i in range(4):
        if (list[i] == 1):
            timage.append((imgs[:, 0:16, 0:16, :]/255.0).reshape(16, 16, 3))
            tlabel.append([1,0,0,0])
        if (list[i] == 2):
            timage.append((imgs[:, 0:16, 16:32, :]/255.0).reshape(16, 16, 3))
            tlabel.append([0,1,0,0])
        if (list[i] == 3):
            timage.append((imgs[:, 16:32, 0:16, :]/255.0).reshape(16, 16, 3))
            tlabel.append([0,0,1,0])
        if (list[i] == 4):
            timage.append((imgs[:, 16:32, 16:32, :]/255.0).reshape(16, 16, 3))
            tlabel.append([0,0,0,1])
    tlabel1 = np.array(tlabel).reshape(-1)
    images_train.append(timage)
    labels_train.append(tlabel1)


images_train = np.array(images_train).transpose(0,1,4,2,3)
np.save('images_train.npy',images_train)
labels_train = np.array(labels_train)
np.save('labels_train.npy',labels_train)

images_test = []
labels_test = []
for imgs, labs in test_set:
    imgs = np.array(imgs)
    timage = []
    list=[1,2,3,4]
    random.shuffle(list)
    tlabel = []
    for i in range(4):
        if (list[i] == 1):
            timage.append((imgs[:, 0:16, 0:16, :]/255.0).reshape(16, 16, 3))
            tlabel.append([1,0,0,0])
        if (list[i] == 2):
            timage.append((imgs[:, 0:16, 16:32, :]/255.0).reshape(16, 16, 3))
            tlabel.append([0,1,0,0])
        if (list[i] == 3):
            timage.append((imgs[:, 16:32, 0:16, :]/255.0).reshape(16, 16, 3))
            tlabel.append([0,0,1,0])
        if (list[i] == 4):
            timage.append((imgs[:, 16:32, 16:32, :]/255.0).reshape(16, 16, 3))
            tlabel.append([0,0,0,1])
    tlabel1 = np.array(tlabel).reshape(-1)
    images_test.append(timage)
    labels_test.append(tlabel1)

    
images_test = np.array(images_test).transpose(0,1,4,2,3)
np.save('images_test.npy',images_test)
labels_test = np.array(labels_test)
np.save('labels_test.npy',labels_test)