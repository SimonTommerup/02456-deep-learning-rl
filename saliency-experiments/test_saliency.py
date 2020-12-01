#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import make_env, Storage, orthogonal_init
import matplotlib.pyplot as plt
from IPython.display import clear_output


class PolicyLogitsHook():
  """
  Establishes a forward hook on policy layer output in a total Policy network.

  The method get_logits() returns a torch.Tensor with 15 logits for
  actions given an observation.

  """
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


def saliency_maps(network, observations):
  raise NotImplementedError

# %%

def saliency_frame(network, storage)
