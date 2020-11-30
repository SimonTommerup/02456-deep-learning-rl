import gym
import procgen
import utils
import initial
import torch
import imageio
import numpy as np
from scipy.ndimage.filters import gaussian_filter

# Hyperparameters
total_steps = 3e6
num_envs = 1
num_levels = 100
num_steps = 256
num_epochs = 3
n_features = 256
batch_size = 512
eps = .2
grad_eps = .5
value_coef = .5
entropy_coef = .01
lr=5e-4
use_backgrounds = False

env = utils.make_env(num_envs, start_level=num_levels, num_levels=num_levels, use_backgrounds=use_backgrounds)
obs = env.reset()

n_features = 256 
encoder = initial.Encoder(env.observation_space.shape[0], n_features)
policy = initial.Policy(encoder, n_features, env.action_space.n)
policy = policy.cuda()

frames = []
total_reward = []

# Evaluate policy
policy.eval()


#%%
obs = env.reset()
print(type(obs))
print(obs.shape)
print(obs[0].shape)
print(obs[0][0].shape)

print(obs[0][0])
print(torch.min(obs[0][0]))

a=torch.ones(size=(1,3,4,4))

print(a)
print("INIT", a.shape)
for idx,_ in enumerate(a[0]):
    for i in range(a.shape[2]):
        for j in range(a.shape[3]):
            a[0][idx][i][j] = 2
a[0][0][0][0] = 5
print(a)
print("AFTER LOOP", a.shape)

amean, _ = torch.mean(a, dim=1)
print("AFTER MEAN", amean.shape)
print(amean)

#print("SF ", sf.shape)...
#SF  (512, 512)
#F  (512, 512, 3)

a = np.ones((4,4,3))
b = np.ones((4,4))*2

print(a[:,:,0])
#print(b.shape)


#%%

frames = []
total_reward = []
for _ in range(512):

    # Use policy
    action, log_prob, value = policy.act(obs)

    # Take step in environment
    obs, reward, done, info = env.step(action)


    total_reward.append(torch.Tensor(reward)) 

    # Render environment and store

    hey = env.render(mode='rgb_array')
    print(type(hey))
    print(hey.shape)

    #frame = (torch.Tensor(env.render(mode='rgb_array'))*255.).byte()
    frame = (torch.Tensor(env.render(mode='rgb_array'))).byte()
    frames.append(frame)

frames = torch.stack(frames)
imageio.mimsave('vid_' + 'no_number' + '.mp4', frames, fps=25)



# %%
print(range(0,8,2))
# %%
d=8
for i in range(0,64+d,d):
    print(i)
# %%
