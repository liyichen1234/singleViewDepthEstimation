import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py
import os
from PIL import Image
f=h5py.File("/data01/lyc/NYUv2/nyu_depth_v2_labeled.mat")
labels=f["labels"]
labels=np.array(labels)
path_converted='/data01/lyc/NYUv2/nyu_labels'
if not os.path.isdir(path_converted):
    os.makedirs(path_converted)
 
labels_number=[]
for i in range(len(labels)):
    labels_number.append(labels[i])
    labels_0=np.array(labels_number[i])
    #print labels_0.shape
    print (type(labels_0))
    label_img=Image.fromarray(np.uint8(labels_number[i]))
    #label_img = label_img.rotate(270)
    label_img = label_img.transpose(Image.ROTATE_270)
 
    iconpath='/data01/lyc/NYUv2/nyu_labels/'+str(i)+'.png'
    label_img.save(iconpath, 'PNG', optimize=True)