import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from utils import make_env, Storage, orthogonal_init
import matplotlib.pyplot as plt
from IPython.display import clear_output
import procgen

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Encoder(nn.Module):
  def __init__(self, in_channels, feature_dim):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
        Flatten(),
        nn.Linear(in_features=1024, out_features=feature_dim), nn.ReLU()
    )
    self.apply(orthogonal_init)

  def forward(self, x):
    return self.layers(x)


class Policy(nn.Module):
  def __init__(self, encoder, feature_dim, num_actions):
    super().__init__()
    self.encoder = encoder
    self.policy = orthogonal_init(nn.Linear(feature_dim, num_actions), gain=.01)
    self.value = orthogonal_init(nn.Linear(feature_dim, 1), gain=1.)

  def act(self, x):
    with torch.no_grad():
      x = x.cuda().contiguous()
      dist, value = self.forward(x)
      action = dist.sample()
      log_prob = dist.log_prob(action)
    
    return action.cpu(), log_prob.cpu(), value.cpu()

  def forward(self, x):
    x = self.encoder(x)
    logits = self.policy(x)
    value = self.value(x).squeeze(1)
    dist = torch.distributions.Categorical(logits=logits)

    return dist, value

num_envs = 1
num_levels = 1
num_features = 256 
use_backgrounds=False


class PolicyHook():
  def __init__(self, net, mode):
    self.name = mode
    self.module = net._modules[self.name]
    self.out = {}
    self._rgsthk(self.module)

  def _rgsthk(self, module):
    self.module.register_forward_hook(self.hook_fn)

  def hook_fn(self, module, input, output):
    self.out[self.name] = output

  def get_logits(self):
    return self.out[self.name][0]

if __name__ == "__main__":
      env = utils.make_env(num_envs, env_name="starpilot", start_level=num_levels, num_levels=num_levels, use_backgrounds=use_backgrounds)
      obs = env.reset()

      encoder = Encoder(env.observation_space.shape[0], num_features)
      policy = Policy(encoder, num_features, env.action_space.n)
      policy.cuda()
      policy.eval()
      phook = PolicyHook(policy, mode="policy")
      vhook = PolicyHook(policy, mode="value")

      # Use policy on observation on frame
      action,_,_ = policy.act(obs)

      # Get logits
      logits = phook.get_logits()
      value = vhook.get_logits()

      print(logits)
      print("LenLogits:", len(logits))

      print(value)
      print("LenLogits:", len(value))
      