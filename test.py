import cv2 
import os
from dataset import CustomData
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

path = "/home/arpan/Desktop/Diffusion/img_align_celeba"
data = CustomData('train', im_path=path)

data_loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=4)
dataiter = iter(data_loader)

image = next(dataiter)
image = image.numpy()
image= np.squeeze(image,0)

# plt.imshow(np.transpose(image,(2,1,0)))
cv2.imshow('jj',np.transpose(image,(2,1,0)))
cv2.waitKey(3)

