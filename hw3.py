import jittor as jt
from jittor import nn, Module, optim, init
import numpy as np
import sys, os
import random
import math 
if jt.has_cuda:
    jt.flags.use_cuda = 1
from jittor.dataset.mnist import MNIST 
import matplotlib.pyplot as plt
import pylab as pl


import gzip
from PIL import Image
from jittor.dataset import Dataset
from jittor_utils.misc import download_url_to_local

class MNIST(Dataset):
    def __init__(self, data_root="./mnist_data/", train=True ,download=True, batch_size=1, shuffle=False):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.is_train = train
        if download == True:
            self.download_url()

        filesname = [
                "train-images-idx3-ubyte.gz",
                "t10k-images-idx3-ubyte.gz",
                "train-labels-idx1-ubyte.gz",
                "t10k-labels-idx1-ubyte.gz"
        ]
        self.mnist = {}
        if self.is_train:
            with gzip.open(data_root + filesname[0], 'rb') as f:
                self.mnist["images"] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28, 28)
            with gzip.open(data_root + filesname[2], 'rb') as f:
                self.mnist["labels"] = np.frombuffer(f.read(), np.uint8, offset=8)
        else:
            with gzip.open(data_root + filesname[1], 'rb') as f:
                self.mnist["images"] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28, 28)
            with gzip.open(data_root + filesname[3], 'rb') as f:
                self.mnist["labels"] = np.frombuffer(f.read(), np.uint8, offset=8)
        assert(self.mnist["images"].shape[0] == self.mnist["labels"].shape[0])
        self.total_len = self.mnist["images"].shape[0]
        self.set_attrs(batch_size = self.batch_size, total_len=self.total_len, shuffle= self.shuffle)
    def __getitem__(self, index):
        img = Image.fromarray (self.mnist['images'][index]) 
        img = np.array (img)
        img = img[np.newaxis, :]
        return np.array((img / 255.0), dtype = np.float32), self.mnist['labels'][index]

    def download_url(self):
        resources = [
            ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
            ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
            ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
            ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
        ]

        for url, md5 in resources:
            filename = url.rpartition('/')[2]
            download_url_to_local(url, filename, self.data_root, md5)

class MNIST_bias(Dataset):
    def __init__(self, data_root="./mnist_data/", train=True ,download=True, batch_size=1, shuffle=False):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.is_train = train
        if download == True:
            self.download_url()
        filesname = [
                "train-images-idx3-ubyte.gz",
                "t10k-images-idx3-ubyte.gz",
                "train-labels-idx1-ubyte.gz",
                "t10k-labels-idx1-ubyte.gz"
        ]
        self.mnist = {}
        if self.is_train:
            with gzip.open(data_root + filesname[0], 'rb') as f:
                self.mnist["imagesf"] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28, 28)
            with gzip.open(data_root + filesname[2], 'rb') as f:
                self.mnist["labelsf"] = np.frombuffer(f.read(), np.uint8, offset=8)
            self.mnist["labels1"] = []
            self.mnist["images1"] = []
            for i in range(len(self.mnist["labelsf"])):
                if i % 10 != 0:
                    if self.mnist["labelsf"][i] >= 5:
                        self.mnist["labels1"].append(self.mnist["labelsf"][i])
                else:
                    self.mnist["labels1"].append(self.mnist["labelsf"][i])
            self.mnist["labels1"] = np.array(self.mnist["labels1"])
            self.mnist["labels"] = []
            self.mnist["images"] = []
            j = 0
            for i in range(len(self.mnist["labelsf"])):
                if i % 10 != 0:
                    if self.mnist["labelsf"][i] >= 5:
                        self.mnist["labels"].append(self.mnist["labelsf"][i])
                        self.mnist["images"].append(self.mnist["imagesf"][i])
                        j += 1
                        if (j == int(len(self.mnist["labels1"])/32)*32): break
                else:
                    self.mnist["labels"].append(self.mnist["labelsf"][i])
                    self.mnist["images"].append(self.mnist["imagesf"][i])
                    j += 1
                    if (j == int(len(self.mnist["labels1"])/32)*32): break
            self.mnist["labels"] = np.array(self.mnist["labels"])
            self.mnist["images"] = np.array(self.mnist["images"])
        else:
            with gzip.open(data_root + filesname[1], 'rb') as f:
                self.mnist["images"] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28, 28)
            with gzip.open(data_root + filesname[3], 'rb') as f:
                self.mnist["labels"] = np.frombuffer(f.read(), np.uint8, offset=8)
        
        assert(self.mnist["images"].shape[0] == self.mnist["labels"].shape[0])
        self.total_len = self.mnist["images"].shape[0]
        self.set_attrs(batch_size = self.batch_size, total_len=self.total_len, shuffle= self.shuffle)
    def __getitem__(self, index):
        img = Image.fromarray (self.mnist['images'][index]) 
        img = np.array (img)
        img = img[np.newaxis, :]
        return np.array((img / 255.0), dtype = np.float32), self.mnist['labels'][index]

    def download_url(self):
        resources = [
            ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
            ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
            ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
            ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
        ]

        for url, md5 in resources:
            filename = url.rpartition('/')[2]
            download_url_to_local(url, filename, self.data_root, md5)

class MNIST_bias_regenerated(Dataset):
    def __init__(self, data_root="./mnist_data/", train=True ,download=True, batch_size=1, shuffle=False):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.is_train = train
        if download == True:
            self.download_url()
        filesname = [
                "train-images-idx3-ubyte.gz",
                "t10k-images-idx3-ubyte.gz",
                "train-labels-idx1-ubyte.gz",
                "t10k-labels-idx1-ubyte.gz"
        ]
        self.mnist = {}
            
        if self.is_train:
            with gzip.open(data_root + filesname[0], 'rb') as f:
                self.mnist["imagesf"] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28, 28)
            with gzip.open(data_root + filesname[2], 'rb') as f:
                self.mnist["labelsf"] = np.frombuffer(f.read(), np.uint8, offset=8)
            self.mnist["labels1"] = []
            self.mnist["images1"] = []
            for i in range(len(self.mnist["labelsf"])):
                if i % 10 != 0:
                    if self.mnist["labelsf"][i] >= 5:
                        self.mnist["labels1"].append(self.mnist["labelsf"][i])
                else:
                    self.mnist["labels1"].append(self.mnist["labelsf"][i])
            self.mnist["labels1"] = np.array(self.mnist["labels1"])
            self.mnist["labelsb"] = []
            self.mnist["imagesb"] = []
            j = 0
            for i in range(len(self.mnist["labelsf"])):
                if i % 10 != 0:
                    if self.mnist["labelsf"][i] >= 5:
                        self.mnist["labelsb"].append(self.mnist["labelsf"][i])
                        self.mnist["imagesb"].append(self.mnist["imagesf"][i])
                        j += 1
                        if (j == int(len(self.mnist["labels1"])/32)*32): break
                else:
                    self.mnist["labelsb"].append(self.mnist["labelsf"][i])
                    self.mnist["imagesb"].append(self.mnist["imagesf"][i])
                    j += 1
                    if (j == int(len(self.mnist["labels1"])/32)*32): break
            self.mnist["labelsb"] = np.array(self.mnist["labelsb"])
            self.mnist["imagesb"] = np.array(self.mnist["imagesb"])
            self.mnist["labels2"] = []
            for i in range(len(self.mnist["labelsb"])):
                if self.mnist["labelsb"][i] <= 4:
                    for j in range(10):
                        self.mnist["labels2"].append(self.mnist["labelsb"][i])
                else:
                    self.mnist["labels2"].append(self.mnist["labelsb"][i])
            self.mnist["labels2"] = np.array(self.mnist["labels2"])
            self.mnist["labels"] = []
            self.mnist["images"] = []
            k = 0
            for i in range(len(self.mnist["labelsb"])):
                if self.mnist["labelsb"][i] <= 4:
                    for l in range(10):
                        self.mnist["labels"].append(self.mnist["labelsb"][i])
                        self.mnist["images"].append(self.mnist["imagesb"][i])
                        k += 1
                        if (k == int(len(self.mnist["labels2"])/32)*32): break
                else:
                    self.mnist["labels"].append(self.mnist["labelsb"][i])
                    self.mnist["images"].append(self.mnist["imagesb"][i])
                    k += 1
                    if (k == int(len(self.mnist["labels2"])/32)*32): break
            self.mnist["labels"] = np.array(self.mnist["labels"])
            self.mnist["images"] = np.array(self.mnist["images"])
        else:
            with gzip.open(data_root + filesname[1], 'rb') as f:
                self.mnist["images"] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28, 28)
            with gzip.open(data_root + filesname[3], 'rb') as f:
                self.mnist["labels"] = np.frombuffer(f.read(), np.uint8, offset=8)
        
        assert(self.mnist["images"].shape[0] == self.mnist["labels"].shape[0])
        self.total_len = self.mnist["images"].shape[0]
        # this function must be called
        self.set_attrs(batch_size = self.batch_size, total_len=self.total_len, shuffle= self.shuffle)
    def __getitem__(self, index):
        img = Image.fromarray (self.mnist['images'][index]) 
        img = np.array (img)
        img = img[np.newaxis, :]
        return np.array((img / 255.0), dtype = np.float32), self.mnist['labels'][index]

    def download_url(self):
        resources = [
            ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
            ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
            ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
            ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
        ]

        for url, md5 in resources:
            filename = url.rpartition('/')[2]
            download_url_to_local(url, filename, self.data_root, md5)
    
class RNN(Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_size):
        super(RNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.rnn = nn.LSTMCell(         
            input_size=self.input_dim,
            hidden_size=self.hidden_dim
        )
        
        self.out = nn.Linear(self.hidden_dim, self.output_dim)

    def execute(self, x):
        hx = jt.zeros(self.batch_size, self.hidden_dim) # (batch, hidden_size)
        cx = jt.zeros(self.batch_size, self.hidden_dim)
        for i in range(x.shape[0]):
            hx, cx = self.rnn(x[i], (hx, cx))
        out = self.out(hx)
        return out



batch_size = 32
learning_rate = 0.1
momentum = 0.9
weight_decay = 1e-4
epochs = 8
losses = []
losses_idx = []

model = RNN(input_dim = 28, hidden_dim = 64, output_dim = 10, batch_size = batch_size)
optimizer = nn.SGD(model.parameters(), learning_rate, momentum, weight_decay)

train_loader = MNIST(train=True, batch_size=batch_size, shuffle=True)
train_loader_bias = MNIST_bias(train=True, batch_size=batch_size, shuffle=True)
train_loader_bias_regenerated = MNIST_bias_regenerated(train=True, batch_size=batch_size, shuffle=True)
val_loader = MNIST(train=False, batch_size=1, shuffle=False)

def train(model, train_loader, optimizer, epoch, losses, losses_idx):
    model.train()
    lens = len(train_loader)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = jt.array(inputs.view(batch_size,28,28)).transpose(1,0,2)
        targets = jt.array(targets.view(-1))
        outputs = model(inputs)
        loss = nn.cross_entropy_loss(outputs, targets)
        optimizer.zero_grad()
        optimizer.step (loss)
        losses.append(loss.numpy()[0])
        losses_idx.append(epoch * lens + batch_idx)
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader) ,
                100. * batch_idx / len(train_loader), loss.numpy()[0]))

def val(model, val_loader):
    model.eval()
    total_acc = 0
    total_num = 0
    acc_small = 0
    acc_big = 0
    num_small = 0
    num_big = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs = jt.array(inputs.view(28,28))
        outputs = model(inputs)
        pred = np.argmax(outputs.numpy(), axis=1)[1]
        acc = np.sum(targets.numpy()==pred)
        if (targets.numpy() <= 4):
            acc_small += acc
            num_small += 1
        else:
            acc_big += acc
            num_big += 1
            
        total_acc += acc
        total_num += 1

    print('Test Small Acc =', acc_small / num_small)
    print('Test Big Acc =', acc_big / num_big)
    print('Test Acc =', total_acc / total_num)

for epoch in range(epochs):
    #train(model, train_loader, optimizer, epoch, losses, losses_idx)
    #train(model, train_loader_bias, optimizer, epoch, losses, losses_idx)
    train(model, train_loader_bias_regenerated, optimizer, epoch, losses, losses_idx)

val(model, val_loader)

model_path = './mnist_model.pkl'
model.save(model_path)
