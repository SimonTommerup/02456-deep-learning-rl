import gym
import procgen
import utils
import initial
import torch
import imageio
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from torchvision.transforms import Resize
import torch.nn.functional as F
import cv2
from tqdm import tqdm
import models
import matplotlib.pyplot as plt



def get_mask(center, size, r):
    y,x = np.ogrid[-center[0]:size[0]-center[0], -center[1]:size[1]-center[1]]
    keep = x*x + y*y <= 1
    mask = np.zeros(size) 
    mask[keep] = 1 # select a circle of pixels
    mask = gaussian_filter(mask, sigma=r) # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
    return mask/mask.max()

def channel_to_numpy(tensor):
    size = tensor.shape[0]
    ndarr = tensor.numpy()
    ndarr = np.reshape(ndarr, (size,size))
    return ndarr

def channel_to_tensor(ndarr):
    size = ndarr.shape[0]
    tensor = torch.from_numpy(ndarr)
    tensor = tensor.view(1,1,size,size)
    return tensor

def gaussian_blur(frame, mask):
    gonc = []
    goncmask = []
    for idx,_ in enumerate(frame):
        channel = frame[0][idx]
        channel = channel_to_numpy(channel)
        channel = channel * (np.ones((64,64)) - mask) + gaussian_filter(channel, sigma=3)*mask
        gonc.append(gaussian_filter(channel, sigma=3))
        goncmask.append(gaussian_filter(channel, sigma=3)*mask)
        channel = channel_to_tensor(channel)
        frame[0][idx] = channel
    return frame, gonc, goncmask

i = 0
j = 0

mask = get_mask(center=[32,32], size=[64,64], r=5)
num_envs = 1
num_levels = 1
num_features = 256 
use_backgrounds=False
env = utils.make_env(num_envs, env_name="starpilot", start_level=num_levels, num_levels=num_levels, use_backgrounds=use_backgrounds)
obs = env.reset()

def saliency_frame(frame, pixel_step):
    ps = pixel_step
    bfs = []
    for idx,_ in enumerate(frame[0]):
        inner_bfs = []
        for i in range(0, frame.shape[2], ps):
            for j in range(0, frame.shape[3], ps):
                mask = get_mask(center=[i,j], size=[64,64], r=5)
                blurred_frame, g, gm = gaussian_blur(frame.clone(), mask)
                inner_bfs.append(blurred_frame)
        bfs.append(inner_bfs)
    return bfs


bfs = saliency_frame(obs, pixel_step=4)

#%%


print(type(bfs[0][0]))

a = torch.stack(bfs[0])







# %%

a.shape
# %%

a[0][0].shape



# %%
plt.imshow(img[0])
plt.imshow(img[1])
plt.imshow(img[2])
plt.show()
# %%
for img in a:
  img = img[0][0].numpy()
  plt.imshow(img[0], cmap=plt.get_cmap("magma"))
  plt.imshow(img[1], cmap=plt.get_cmap("magma"))
  plt.imshow(img[2], cmap=plt.get_cmap("magma"))
  plt.show()

# %%
