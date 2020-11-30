import gym
import procgen
import utils
import initial
import torch
import imageio
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import cv2
from tqdm import tqdm
import models

class PolicyLogitsHook():
  def __init__(self, net):
    self.name = "policy"
    self.module = net._modules[self.name]
    self.out = {}
    self._rgsthk(self.module)

  def _rgsthk(self, module):
    self.module.register_forward_hook(self.hook_fn)

  def hook_fn(self, module, input, output):
    self.out[self.name] = output

  def get_logits(self):
    return self.out[self.name][0]


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

def upsample_saliency_frame(saliency_frame):
    LO_RES_SIZE = 64
    HI_RES_SIZE = 512
    ndarr = saliency_frame.numpy()
    ndarr = np.reshape(ndarr, (LO_RES_SIZE,LO_RES_SIZE))
    # resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    ndarr = cv2.resize(ndarr, dsize=(HI_RES_SIZE,HI_RES_SIZE), interpolation=cv2.INTER_LINEAR) #.astype(np.float32)
    return ndarr
def red_frame():
    rf = np.ones((512,512))
    rf = rf * 255.
    return rf

def gaussian_blur(frame, mask):
    for idx,_ in enumerate(frame):
        channel = frame[0][idx]
        channel = channel_to_numpy(channel)
        channel = channel * (np.ones((64,64)) - mask) + gaussian_filter(channel, sigma=3)*mask
        channel = channel_to_tensor(channel)
        frame[0][idx] = channel
    return frame

def saliency_score(x,y):
    return torch.mean(torch.abs(x-y)**2)

def saliency_frame(net, hook, logits, frame, pixel_step):
    ps = pixel_step
    saliency_frame = torch.zeros(size=(1,3,64,64))
    for idx,_ in enumerate(saliency_frame[0]):
        for i in range(0, saliency_frame.shape[2] + ps, ps):
            for j in range(0, saliency_frame.shape[3] + ps, ps):
                mask = get_mask(center=[i,j], size=[64,64], r=5)
                blurred_frame = gaussian_blur(frame, mask)

                _, _,_ = net.act(blurred_frame)
                blurred_logits = hook.get_logits()

                score = saliency_score(logits, blurred_logits)
                saliency_frame[0][idx][int(i/ps)][int(j/ps)] = score

    return saliency_frame

def saliency_mode(saliency_frame, mode):
    assert(mode in ["mean", "max"], "Mode must be either \"mean\" or \"max\"")
    if mode == "mean":
        saliency_frame = saliency_frame.mean(dim=1)
    elif mode == "max":
        saliency_frame, _ = torch.max(saliency_frame, dim=1)
    return saliency_frame

num_envs = 1
num_levels = 1
num_features = 256 
use_backgrounds=False

if __name__ == "__main__":
    env = utils.make_env(num_envs, start_level=num_levels, num_levels=num_levels, use_backgrounds=use_backgrounds)
    obs = env.reset()
    """
    encoder = models.DQNEncoder(env.observation_space.shape[0], num_features)
    policy = models.Policy(encoder, num_features, env.action_space.n)
    policy.load_state_dict(torch.load("2_500_lvls/temp_model.pt"))
    policy.cuda()
    policy.eval()
    """
    encoder = initial.Encoder(env.observation_space.shape[0], num_features)
    policy = initial.Policy(encoder, num_features, env.action_space.n)
    policy.cuda()
    policy.eval()

    hook = PolicyLogitsHook(policy)

    frames = []
    for _ in tqdm(range(16)):

        # Use policy on observation on frame
        action,_,_ = policy.act(obs)

        # Get logits
        logits = hook.get_logits()

        # Get saliency frame
        sf = saliency_frame(net=policy, hook=hook, logits=logits, frame=obs, pixel_step=4)
        sf = saliency_mode(sf, mode="mean")
        sf = upsample_saliency_frame(sf)
        print("S MAX:" , np.max(sf))
        print("S MIN:" , np.max(sf))

        # frame of 255s all over, slight proof of concept :-)
        rf = red_frame()

        # Rendering
        frame = env.render(mode="rgb_array")
        print("F MAX:" , np.max(frame))
        print("F MIN:" , np.max(frame))

        #frame[:,:,0]= frame[:,:,0] + sf
        frame[:,:,0]= frame[:,:,0] + rf

        # Record frame to frames stack
        frame = (torch.Tensor(frame)).byte()
        frames.append(frame)

        # Take step in environment
        obs,_,_,_ = env.step(action)

    frames = torch.stack(frames)
    imageio.mimsave('vid_' + 'slow_saliency_test' + '.mp4', frames, fps=25)
    
    print(policy.state_dict)
