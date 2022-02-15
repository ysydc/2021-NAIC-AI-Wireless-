#!/usr/bin/env python3
"""An Implement of an autoencoder with pytorch.
This is the code for 2020 NIAC https://naic.pcl.ac.cn/.
"""

import numpy as np
import torch.nn as nn
import scipy.io as scio
import os
import h5py
import torch
import math
from Model_define_pytorch import AutoEncoder, DatasetFolder, NMSE, Score


# Parameters for training
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
use_single_gpu = True  # select whether using single gpu or multiple gpus
torch.manual_seed(1)

batch_size = 64
num_workers = 4

# load train data
data_load_address = './data'
mat = scio.loadmat(data_load_address+'/Htrain.mat')
data_train = mat['H_train']  # shape=8000*126*128*2
data_train = np.transpose(data_train.astype('float32'),[0,3,1,2])
print(np.shape(data_train))

# load test data
mat = scio.loadmat(data_load_address+'/Htest.mat')
data_test = mat['H_test']  # shape=2000*126*128*2
data_test = np.transpose(data_test.astype('float32'),[0,3,1,2])
print(np.shape(data_test))

# put two thirds of test data into train data
# nd = math.floor(data_test.shape[0]*2/3)
nd = 1000
ftest = data_test[:nd]
xtest = data_test[nd:]
xtrain = np.concatenate((data_train, ftest), axis=0)
print(np.shape(xtrain))
print(np.shape(xtest))

# dataLoader for training
train_dataset = DatasetFolder(xtrain)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

# dataLoader for training
test_dataset = DatasetFolder(xtest)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

print('Data is ready.')

# parameters for data
feedback_bits = 512
learning_rate = 1e-3
epochs = 500
print_freq = 100  # print frequency (default: 60)

# Model construction
model = AutoEncoder(feedback_bits)
if use_single_gpu:
    model = model.cuda()
else:
    # DataParallel will divide and allocate batch_size to all available GPUs
    model = torch.nn.DataParallel(model).cuda()

criterion = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print('Start training...')
score_best = -1

for epoch in range(epochs):
    print('-----------------------epoch {0}  -----------------------'.format(epoch))
    total_loss = 0
    # model training
    model.train()
    for i, xinput in enumerate(train_loader):
        xinput = xinput.cuda()
        # compute output
        output = model(xinput)
        loss = criterion(output, xinput)
        total_loss += loss
        
        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.8f}\t'.format(
                epoch, i, len(train_loader), loss=loss.item()))
    
    average_loss = total_loss/len(train_loader)
    print('The average loss is {:.8f}'.format(average_loss))
    
    # model evaluating
    model.eval()
    with torch.no_grad():
        for i, xinput in enumerate(test_loader):
            xinput = xinput.cuda()
            output = model(xinput)
            output = output.cpu().numpy()
            if i == 0:
                y_test = output
            else:
                y_test = np.concatenate((y_test, output), axis=0)
        
        # need convert channel first to channel last for evaluate.
        NMSE_test = NMSE(np.transpose(xtest, (0, 2, 3, 1)), np.transpose(y_test, (0, 2, 3, 1)))
        print('The NMSE is ', NMSE_test)
        
        scr = Score(NMSE_test)
        if scr < 0:
            scr=0
        else:
            scr=scr
        print('score=', scr)
       
        if scr > score_best:           
            # model save
            # save encoder
            encoderSave = './Modelsave/encoder.pth.tar'
            torch.save({'state_dict': model.encoder.state_dict(), }, encoderSave)
            # save decoder
            decoderSave = './Modelsave/decoder.pth.tar'
            torch.save({'state_dict': model.decoder.state_dict(), }, decoderSave)
            print("Model saved")
            score_best = scr 

print('score_best =', score_best)