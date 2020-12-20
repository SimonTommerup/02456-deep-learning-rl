import procgen
import cv2
import imageio
import torch
import os
import numpy as np
import models
import utils
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm

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

def gaussian_blur(frame, mask):
    for idx,_ in enumerate(frame):
        channel = frame[0][idx]
        channel = channel_to_numpy(channel)
        channel = channel * (np.ones((64,64)) - mask) + gaussian_filter(channel, sigma=3)*mask
        channel = channel_to_tensor(channel)
        frame[0][idx] = channel
    return frame

def saliency_score(x,y):
    return 0.5*torch.sum((x-y)**2)

def saliency_frame(net, hook, logits, frame, pixel_step):
    ps = pixel_step
    saliency_frame = torch.zeros(size=(1,3,int(64/ps),int(64/ps)))

    for idx,_ in enumerate(frame[0]):
        for i in range(0, frame.shape[2], ps):
            for j in range(0, frame.shape[3], ps):
                mask = get_mask(center=[i,j], size=[64,64], r=5)
                blurred_frame = gaussian_blur(frame.clone(), mask)

                _, _,_ = net.act(blurred_frame)
                blurred_logits = hook.get_logits()
                score = saliency_score(logits, blurred_logits)

                saliency_frame[0][idx][int(i/ps)][int(j/ps)] = score
    
    saliency_frame = F.interpolate(saliency_frame, size=(64,64), mode="bilinear")
    return saliency_frame

def saliency_mode(saliency_frame, mode):
    if mode == "mean":
        saliency_frame = saliency_frame.mean(dim=1)
    elif mode == "max":
        saliency_frame, _ = torch.max(saliency_frame, dim=1)
    return saliency_frame

def saliency_on_procgen(procgen_frame, saliency_frame, channel, constant, sigma=0):
    sf = saliency_frame.numpy()
    sfmax = sf.max()
    sf = cv2.resize(sf[0], dsize=(512,512), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    if sigma==0:
        sf = sf
    else:
        sf = gaussian_filter(sf, sigma)
    
    sf -= sf.min()
    sf = constant * sfmax * sf / sf.max()
    procgen_frame[:,:,channel] += sf.astype("uint64") 
    procgen_frame = procgen_frame.clip(1,255).astype("uint8") 
    return procgen_frame

def color_to_channel(color):
    colors = ["red", "green", "blue"]
    assert color in colors, "Color must be red, green or blue (RGB)"
    channel = [idx for idx, elem in enumerate(colors) if color==elem][0]
    return channel

def get_full_path(model_name):
    _cwd = os.getcwd()
    os.chdir("../experiments")
    _mdir = os.path.join(os.getcwd(), model_name)
    _modelpath = os.path.join(_mdir, "model_" + model_name + ".pt")
    os.chdir(_cwd)
    return _modelpath

# Settings
env_name = "starpilot"
num_envs = 1
start_level = 1
num_levels = 1
num_features = 256 
use_backgrounds=False


if __name__ == "__main__":
    
    env = utils.make_env(num_envs, env_name=env_name, start_level=start_level, num_levels=num_levels, use_backgrounds=use_backgrounds)
    obs = env.reset()

    # NOTE: 
    # Comment in the right encoder:
    # MODEL 2 = DQN
    # MODEL 5 = Impala

    encoder = models.DQNEncoder(env.observation_space.shape[0], num_features)
    #encoder = models.ImpalaModel(env.observation_space.shape[0], num_features)
    policy = models.Policy(encoder, num_features, env.action_space.n)

    #model_folder = "5_500_lvls_impala_valclip"
    model_folder = "0_baseline"
    model_path = get_full_path(model_folder)
    policy.load_state_dict(torch.load(model_path))
    policy.cuda()
    policy.eval()

    hook = PolicyLogitsHook(policy)

    frames = []
    for _ in tqdm(range(128)):

        # Use policy on observation on frame
        action,_,_ = policy.act(obs)

        # Get logits
        logits = hook.get_logits()

        # Get saliency
        sf = saliency_frame(net=policy, hook=hook, logits=logits, frame=obs, pixel_step=4)

        # Set saliency mode
        mode = "max"
        sf = saliency_mode(sf, mode=mode)

        # Rendering
        frame = env.render(mode="rgb_array")

        constant = 5
        sigma = 0
        channel = color_to_channel("red")
        frame = saliency_on_procgen(frame, sf, channel=channel, constant=constant, sigma=sigma)

        # Record frame to frames stack
        frame = (torch.Tensor(frame)).byte()
        frames.append(frame)

        # Take step in environment
        obs,_,_,_ = env.step(action)

    frames = torch.stack(frames)
    video_path = env_name + "_" + model_folder + "_" + f"c={constant}_" + f"sig={sigma}_"+ f"mode={mode}" + ".mp4"
    imageio.mimsave(video_path, frames, fps=5)


