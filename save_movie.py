import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import imageio
import json
import matplotlib.pyplot as plt
from utils import make_env, Storage, orthogonal_init
from IPython.display import clear_output
from utils import make_env, Storage, orthogonal_init
from tqdm import tqdm

from models import ImpalaModel, DQNEncoder, Policy

### Prepare parameters


name = '9_model_5_coinrun_03e8_steps'
#game = 'starpilot'
#game = 'bossfight'
game = "coinrun"

total_steps = 8e6
num_envs = 32
num_steps = 512
num_epochs = 3
n_features = 256
batch_size = 512
eps = .2
grad_eps = .5
value_coef = .5
entropy_coef = .01
lr=5e-4
use_backgrounds = False
save_step = 5e5
normalize_reward = True

# Experiments
use_impala = True
num_levels = 500
use_clipped_value = True
penalize_on_death = False


# Training and validation env
print("Creating environments")
env = make_env(num_envs, env_name=game, num_levels=num_levels, use_backgrounds=use_backgrounds)
env_val = make_env(num_envs, env_name=game, start_level=num_levels, num_levels=10,  use_backgrounds=use_backgrounds)

# Define models
print("Creating model")
if use_impala:
    encoder = ImpalaModel(env.observation_space.shape[0], n_features)
else:
    encoder = DQNEncoder(env.observation_space.shape[0], n_features)
policy = Policy(encoder, n_features, env.action_space.n)
print("Loading model")


# Set values for comparison
path = os.getcwd()
model_dir = name + "/model_" + name + ".pt"
print("PATH IS: ", path)
print("Model dir: ", model_dir)


policy.load_state_dict(torch.load(model_dir))
policy.cuda()




################## Save movie


obs = env_val.reset()

frames = []
total_reward = []

# Evaluate policy
policy.eval()
                
for _ in tqdm(range(num_steps)):

    # Use policy
    action, log_prob, value = policy.act(obs)

    # Take step in environment
    obs, reward, done, info = env_val.step(action)
    total_reward.append(torch.Tensor(reward))  

    # Render environment and store
    frame = (torch.Tensor(env_val.render(mode='rgb_array'))*255.).byte()
    frames.append(frame)

# Calculate average return
total_reward = torch.stack(total_reward).sum(0).mean(0)
print('Average return:', total_reward)

# Save frames as video
frames = torch.stack(frames)
imageio.mimsave('vid_' + name + '.mp4', frames, fps=25)

