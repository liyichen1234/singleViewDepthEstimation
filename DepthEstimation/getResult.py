import os
import cv2
import matplotlib.pyplot as plt
path = "/data01/lyc/CV-Ex2/generated_img"
d_path = "/data01/lyc/CV-Ex2/result/"
image = sorted(os.listdir(path))
for i in range(0,len(image),2):
    path1 = path+"/"+image[i]
    path2 =  path+"/"+image[i+1]
    f, axs = plt.subplots(1, 2, figsize=(16, 9))
    f.tight_layout()
    axs[0].imshow(cv2.cvtColor(cv2.imread(path2),cv2.COLOR_BGR2RGB), cmap='gray')
    axs[0].set_title('Picture', fontsize=18)
    axs[1].imshow(cv2.cvtColor(cv2.imread(path1),cv2.COLOR_BGR2RGB), cmap='gray')
    axs[1].set_title('Depth', fontsize=18)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.savefig(d_path+str(i)+'.jpg')
