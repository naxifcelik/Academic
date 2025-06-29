import pickle
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import torch
import dnnlib
import torch_utils
with open("/home/nax/Desktop/hw3/stylegan3-t-ffhq-1024x1024.pkl","rb") as f :
    a = pickle.load(f)

gan = a["G_ema"]
gan.eval()
for param in gan.parameters():
    param.requires_grad = False
z = torch.randn(1, 512)
img = gan(z,0).numpy().squeeze()
img = np.transpose(img,(1,2,0))
img[img >1] = 1
img[img<-1] = -1
img = 255*(img+1)/2
cv2.imwrite('test.png',img[:,:,[2,1,0]])