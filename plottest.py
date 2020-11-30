def plot_activations(activation, layer):
  for idx in range(len(activation)):
    activation[idx] = activation[idx].cpu()
  act = activation[layer].squeeze()
  fig, axarr = plt.subplots(act.size(0))
  for idx in range(act.size(0)):
      axarr[idx].imshow(act[idx])

# Register hooks on Conv2d layers of encoder
# to a dict storing the activations.
def register_hooks_to_activation(policy, forward=True):
  activation = {}
  def get_activation(name): 
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
  _module = policy.encoder
  name = 0
  for layer in _module.layers:
    if isinstance(layer, nn.Conv2d):
      if forward:
        layer.register_forward_hook(get_activation(name))
      elif not forward:
        layer.register_backward_hook(get_activation(name))
      name += 1
  return activation
activation = register_hooks_to_activation(policy)