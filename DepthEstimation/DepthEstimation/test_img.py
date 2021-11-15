import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import torch.nn as nn
import cv2

from UtilityTest import DepthDataset
from UtilityTest import ToTensor

#location of images
loc_img="/data01/lyc/CV-Ex2/data/nyu2_test"

depth_dataset = DepthDataset(root_dir=loc_img)
depth_dataset = DepthDataset(root_dir=loc_img,transform=transforms.Compose([ToTensor()]))
batch_size=1
train_loader=torch.utils.data.DataLoader(depth_dataset, batch_size)
dataiter = iter(train_loader)
images = dataiter.next()

#importing the model 
from Mobile_model import Model

model = Model().cuda()
model = nn.DataParallel(model)
#load the trained model
model.load_state_dict(torch.load('/data01/lyc/CV-Ex2/data/14.pth'))
model.eval()
# print(model.eval())

#Upscaling image and saving the image
os.mkdir('/data01/lyc/CV-Ex2/generated_img')
for i,sample_batched1 in enumerate(train_loader):
    try:
        image1 = torch.autograd.Variable(sample_batched1['image'].cuda())
        outtt=model(image1 )
        x=outtt.detach().cpu().numpy()
        img=x.reshape(240,320)
        scale_percent = 200 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        plt.imsave('/data01/lyc/CV-Ex2/generated_img/%d_depth.jpg' %i, resized, cmap='inferno') 
        s_img=sample_batched1['image'].detach().cpu().numpy().reshape(3,480,640).transpose(1,2,0)
        plt.imsave('/data01/lyc/CV-Ex2/generated_img/%d_image.jpg' %i, s_img) 
    except:
        continue
