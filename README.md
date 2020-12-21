# Visualizing Improvements of Reinforcement Learning Models
*Featuring Saliency Maps*

This project is carried out as a part of the DTU course "Deep Learning - 02456".

## Motivation for project
In recent years reinforcement learning (RL) models have shown
promising results in scenarios where a series of decisions have to be
made in an unknown environment, e.g. an Atari game, where the input
consists of the raw pixels. For this reason, RL models are often
explored in the setting of computer games.
However, RL currently relies on both sensitive parameter tuning and
lots of computer power, and often the models do not generalize to other
settings.
In this project different RL models are explored and compared in a ‘low
GPU’ setting, to find a good model, while getting an intuition for why it
is good. The starting point is a PPO algorithm and used in Procgen
environments. 

Different models are explored by starting with a simple model and in
turn fine tuning each parameter. Through this approach it is assumed
that the parameters are independent of each other, which might lead to
a suboptimal model but it makes the model exploration much cheaper.
To aid this exploration, the gradients of the encoder network have been
used to visualize which pixels play a role in the model.

## About this repo
This repo is an implementation of the PPO algorithm from this [paper](https://arxiv.org/pdf/1707.06347.pdf) and saliency maps based on this [paper](https://arxiv.org/pdf/1711.00138.pdf)


### Dependencies
A couple of dependencies you might want to be aware of, when running this project:
- GPUs supporting CUDA with 16GB RAM (less might be sufficient)
- torch 1.7.0
- procgen 0.10.4
- gym 0.17.3

You can make a virtual environment and run `pip install -r src/requirements.txt` to install all dependencies. 

### Training
The training code is located in ``src/train.py``. You can play around with the different variables and hyperparameters in the beginning of the script to reproduce our results. The script is not yet fully optimised for others to easily use this, as this way of structuring the code made our experiments more easy. However, it should be fairly easy to navigate in the script. Before you run the ``train.py``, then:
- change ``name``to a desired name for your experiment. Your experiment will be saved to the ``experiments``folder.
- set ``comparison_name`` to a folder name in ``experiments`` to compare rewards in a plot during training 
- change hyper parameters in the beginning of the script to your desired ones

We ran the code through an available DTU server using ``jobscript.sh``. You must add an e-mail to receive the results. **Please run the code from the root folder.**

### Evaluation
Note that you can use one of our pretrained models to evalute. Find these in the experiment folder. Each model is associated with a pdf showing the training performance (see section below). The best performing model in the starpilot environment is located in experiment no. 5. It acheives a validation reward of ~19 and a video including its attention (using saliency maps) can be seen here: https://youtu.be/BVxv4tF7RNg

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/BVxv4tF7RNg/0.jpg)](https://youtu.be/BVxv4tF7RNg)

<iframe width="560" height="315" src="https://www.youtube.com/embed/BVxv4tF7RNg" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

You can visit **NOTEBOOK COMING SOON** to run a model for a couple of steps. 


### Overview of experiments
A couple of experiments are included in the ``experiments`` folder. The naming convention is as follows:
- ``0_baseline`` is our initial model point. We used the Nature-DQN encoder, with 32 environments, 3 epochs, 256 output features from the encoder, batch size 512, entropy coefficient 0.01, value clipping (eps) 0.02, lr=5e-3, gamma (discount reward) = 0.99, num_levels=10 and the PPO implementation from the original paper.
- ``1_100_lvls`` refers to the exact same experiment as above, but where number of lvls is changed to 100.
- ``5_500_lvls_impala_valClip`` is the same as ``0_baseline``, but where the encoder is changed to impala, we now use 500 levels and we expanded the PPO algorithm to also use clipping on the contribution for the value clipping. 
- ``7_model_5_boss_fight``refers to experiment number 7 with the exact same model as in experiment 5 (the one above: ``5_500_lvls_impala_valClip``). 
- etc...

If you want to know more about the experiments, check out our paper!


### Architectures of Models
The different implemented models used for policy, encoder and value function can be found in ``src/models.py``.


### Plotting reward data
For all conducted experiments with a folder, you can plot and compare them using the ``load_and_plot_rewards.py`` to compare performances. Use an interactive python kernel for this!



### Producing videos with saliency maps
`src/saliency.py` allows recording videos with saliency scores added to one of the color channels. 

Change the variable `env_name` to one of the games in the Procgen suite. Examples are "starpilot", "coinrun" and "bigfish". 

In the script itself you will need to 

1. Comment in the appropriate encoder with is dependent on which model will be in the video. As stated above model 5's will use the Impala encoder, where as model 2's will use the DQN encoder. 

2. Change the variable `model_folder` to one of the folders containing the trained model that will be in the video. 

You can control which color channel the saliency maps should be added to by changing the variable `channel` by given input either "red", "green" or "blue" to the function `color_to_channel`. Also you are able to control the scaling `constant` that controls the color intensity and the value `sigma` controlling the value of the standard deviation of an optional Gaussian filter on the frames during the creation of the video. If `sigma = 0` then no Gaussian filter is used. Finally you can adjust if you want the saliency scores to be computed as the mean or max across the color channels by setting `mode` equal to `mean` or `max`.

Your video will be saved to `src` as `env_name` + `model_folder` along with a specificiation of the settings used to make the video. If you use the same game, model and settings twice be advised that the old video will be replaced by the new unless you customize the variable `video_path` in the bottom of the script.

## Acknowledgements
It should be mentioned that train.py is based on a notebook created by Nicklas Hansen - TA in the Deep Learning Course at DTU, 02456.
