import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.utils import shuffle
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
from DepthData_mob import DepthDataset
from DepthData_mob import Augmentation
from DepthData_mob import ToTensor

traincsv=pd.read_csv('/data01/lyc/CV-Ex2/data/nyu2_train.csv')
traincsv = traincsv.values.tolist()
traincsv = shuffle(traincsv, random_state=2)

#display a sample set of image and depth image
depth_dataset = DepthDataset(traincsv=traincsv,root_dir='/data01/lyc/CV-Ex2/')
#loading the mobilNetDepth model
from Mobile_model import Model

import cv2
import kornia

def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
    ssim = kornia.losses.SSIMLoss(window_size=11,max_val=val_range,reduction='none')
    return ssim(img1, img2)

import matplotlib
import matplotlib.cm
import numpy as np

def DepthNorm(depth, maxDepth=1000.0): 
    return maxDepth / depth

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
    value = value.cpu().numpy()[0,:,:]
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin!=vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        value = value*0.
    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value,bytes=True) # (nxmx4)
    img = value[:,:,:3]
    return img.transpose((2,0,1))

def LogProgress(model, writer, test_loader, epoch):
    model.eval()
    sequential = test_loader
    sample_batched = next(iter(sequential))
    image = torch.autograd.Variable(sample_batched['image'].cuda())
    depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
    if epoch == 0: writer.add_image('Train.1.Image', vutils.make_grid(image.data, nrow=6, normalize=True), epoch)
    if epoch == 0: writer.add_image('Train.2.Depth', colorize(vutils.make_grid(depth.data, nrow=6, normalize=False)), epoch)
    output = DepthNorm( model(image) )
    writer.add_image('Train.3.Ours', colorize(vutils.make_grid(output.data, nrow=6, normalize=False)), epoch)
    writer.add_image('Train.3.Diff', colorize(vutils.make_grid(torch.abs(output-depth).data, nrow=6, normalize=False)), epoch)
    del image
    del depth
    del output
    
import time
import datetime

import torch
import torch.nn as nn
import torchvision.utils as vutils    

model = Model().cuda()
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model,device_ids=[0])
#load trained model if needed
print('Model created.')

epochs=15
lr=0.0001
batch_size=16

depth_dataset = DepthDataset(traincsv=traincsv, root_dir='/data01/lyc/CV-Ex2/',
                transform=transforms.Compose([Augmentation(0.5),ToTensor()]))
train_loader=DataLoader(depth_dataset, batch_size, shuffle=True)
l1_criterion = nn.L1Loss()

optimizer = torch.optim.Adam(model.parameters(), lr)

# Start training...
for epoch in range(epochs):
    path='/data01/lyc/CV-Ex2/data/'+str(epoch)+'.pth'        
    torch.save(model.state_dict(), path)
    batch_time = AverageMeter()
    losses = AverageMeter()
    N = len(train_loader)
    model.train()
    end = time.time()

    for i, sample_batched in enumerate(train_loader):
        optimizer.zero_grad()
        #Prepare sample and target
        image = torch.autograd.Variable(sample_batched['image'].cuda())
        depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
        # Normalize depth
        depth_n = DepthNorm( depth )
        # Predict
        output = model(image)
        # Compute the loss
        l_depth = l1_criterion(output, depth_n)
        l_ssim = torch.clamp((1 - ssim(output, depth_n, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)
        loss = (1.0 * l_ssim.mean().item()) + (0.1 * l_depth)
        # Update step
        losses.update(loss.data.item(), image.size(0))
        loss.backward()
        optimizer.step()
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        eta = str(datetime.timedelta(seconds=int(batch_time.val*(N - i))))
        # Log progress
        niter = epoch*N+i
        if i % 5 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
            'ETA {eta}\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})'
            .format(epoch, i, N, batch_time=batch_time, loss=losses, eta=eta))
    path='/data01/lyc/CV-Ex2/'+str(epoch)+'.pth'        
    torch.save(model.state_dict(), path)    
#Evaluations

model = Model().cuda()
model = nn.DataParallel(model)
#load the model if needed
model.eval()
batch_size=1

depth_dataset = DepthDataset(traincsv=traincsv, root_dir='/data01/lyc/CV-Ex2/',
                transform=transforms.Compose([Augmentation(0.5),ToTensor()]))
train_loader=DataLoader(depth_dataset, batch_size, shuffle=True)
for sample_batched1  in (train_loader):
    image1 = torch.autograd.Variable(sample_batched1['image'].cuda())
    outtt=model(image1 )
    break
    #ploting the evaluated images

x=outtt.detach().cpu().numpy()
x.shape
x=x.reshape(240,320)
plt.figure()
plt.imsave(sample_batched1['image'].detach().cpu().numpy().reshape(3,480,640).transpose(1,2,0),"evaluate.jpg")