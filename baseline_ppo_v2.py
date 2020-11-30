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


def save_movie(policy, env_val):
    obs = env_val.reset()

    frames = []
    total_reward = []

    # Evaluate policy
    policy.eval()
                    
    for _ in tqdm(range(512)):

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




#_______________________________ SET MODELS AND HYPERPARAMETERS _______________________________

name = 'random_action_baseline_v2'

total_steps = 10e6
num_envs = 16
num_levels = 5
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
normalize_reward = False
use_impala = True


# Training and validation env
env = make_env(num_envs, num_levels=num_levels, use_backgrounds=use_backgrounds)
env_val = make_env(num_envs, start_level=num_levels, num_levels=10,  use_backgrounds=use_backgrounds)

# Define models
if use_impala:
    encoder = ImpalaModel(env.observation_space.shape[0], n_features)
else:
    encoder = DQNEncoder(env.observation_space.shape[0], n_features)
policy = Policy(encoder, n_features, env.action_space.n)
policy.cuda()
optimizer = torch.optim.Adam(policy.parameters(), lr=lr, eps=1e-5) # CHANGED lr=5e-4, 1e-3

# Set baseline values for comparison
mean_rew_baseline = [3.75, 4.46875, 4.3125, 4.25, 4.71875, 4.09375, 6.375, 4.46875, 4.25, 5.34375, 5.5, 6.84375, 8.0, 7.25, 6.15625, 7.9375, 8.15625, 8.40625, 7.8125, 8.1875, 7.625, 9.6875, 8.8125, 8.3125, 9.9375, 10.2812, 8.375, 9.78125, 10.5312, 8.59375, 9.96875, 10.1562, 9.625, 9.8125, 9.71875, 8.5625, 11.125, 12.4687, 11.4062, 10.4687, 10.2187, 11.5937, 9.9375, 11.75, 12.0625, 11.375, 12.8125, 11.25, 13.4687, 11.875, 12.2812, 11.5937, 13.0625, 11.9375, 13.125, 12.7812, 11.8437, 11.8437, 13.0937, 13.2187, 14.0, 12.5625, 11.625, 13.0625, 13.4687, 12.6875, 12.125, 13.5, 13.9062, 12.4062, 14.1875, 12.9687, 13.4375, 13.1875, 14.4062, 14.4062, 14.3125, 13.25, 12.6562, 13.7187, 13.25, 13.875, 13.375, 13.3437, 12.2812, 14.4062, 14.5312, 14.125, 15.1562, 13.1875, 14.4375, 14.25, 13.9687, 13.8437, 14.1875, 14.875, 14.1875, 15.0312, 14.0, 15.8437, 14.75, 14.4687, 14.0625, 14.3125, 14.7187, 14.8437, 13.125, 14.5937, 15.2812, 13.875, 14.1875, 14.9062, 13.9062, 13.125, 15.125, 13.7812, 14.5625, 13.2812, 13.3125, 14.9687, 13.9375, 15.4687, 13.625, 15.7187, 14.9062, 14.7187, 14.1562, 14.4375, 14.2187, 13.9687, 16.0, 15.1562, 16.0937, 14.5312, 15.9062, 14.2812, 16.4062, 15.6875, 16.375, 14.8125, 14.75, 15.625, 15.9687, 15.1562, 15.125, 16.3437, 15.75, 16.8437, 15.7812, 16.4062, 15.6562, 15.9687, 16.375, 16.0625, 15.0625, 15.4062, 16.0625, 15.875, 16.1875, 16.0937, 16.3437, 15.75, 17.2812, 16.8437, 15.75, 16.2812, 14.8125, 15.75, 16.4687, 17.875, 18.25, 16.0625, 15.1562, 15.9687, 17.875, 15.9687, 16.1562, 18.0625, 17.625, 16.75, 16.3437, 17.1562, 16.0937, 17.0625, 16.2187, 18.0937, 17.1562, 17.125, 16.125, 17.9687, 16.7812, 17.4062, 16.5937, 16.5625, 17.0312, 16.625, 16.3437, 16.9687, 15.8125, 16.3437, 15.9062, 16.0312, 17.5625, 15.9375, 18.0937, 17.1562, 16.7187, 17.4375, 14.9687, 16.3437, 16.125, 17.0937, 16.0, 16.25, 17.7187, 16.1562, 15.5312, 16.5625, 16.7812, 17.3125, 17.4062, 18.875, 14.3437, 17.375, 16.8125, 16.3125, 17.1562, 18.375, 17.75, 18.2187, 16.875, 17.6562, 17.3437, 17.9687, 18.5312, 16.8125, 17.75, 17.2187, 16.3125, 17.3125, 18.7812, 17.0, 17.9062, 15.375, 16.5312, 16.4062, 16.2187, 17.5312, 16.9687, 17.3125, 17.0312, 16.5625, 17.1562, 16.7187, 18.0625, 17.0937, 19.0625, 17.1875, 16.5625, 17.75, 16.75, 16.3437, 17.0625, 16.875, 16.5, 17.0937, 18.2187, 16.1875, 16.7187, 16.875, 17.7812, 16.2812, 15.9375, 17.6875, 16.9062, 17.375, 16.375, 17.3437, 17.9687, 16.8125, 18.1562, 17.7812, 17.7187, 18.2187, 17.5, 17.5937, 17.875, 17.625, 16.625, 18.25, 17.4375, 17.4062, 18.2812, 17.9687, 18.2187, 18.0312, 17.1875, 15.2812, 18.5, 18.0, 16.625, 17.625, 17.4687, 16.9375, 16.8125, 17.3125, 16.875, 17.6562, 17.4375, 17.875, 17.375, 16.2187, 17.2812, 17.125, 17.8125, 17.0, 17.4375, 18.7812, 16.6875, 18.25, 18.0, 18.625, 19.0625, 18.0937, 18.5937, 16.7187, 16.5625, 17.5625, 15.1875, 17.5625, 17.1562, 18.3125, 17.1875, 16.7812, 17.9375, 15.6875, 16.8437, 15.9375, 18.0937, 16.9062, 17.2812, 16.5625, 16.4687, 16.8437, 15.75, 16.5625, 17.0625, 18.2187, 16.9687, 17.375, 16.0625, 18.1875, 18.375, 17.6562, 17.5312, 15.5625, 17.125, 18.8125, 17.3125, 16.125, 16.5937, 18.0, 16.5, 18.5937, 17.8125, 17.1875, 18.7812, 16.8437, 16.2812, 17.875, 15.6875, 18.0937, 18.0937, 15.8125, 19.375, 19.1875, 16.0312, 17.625, 17.5312, 17.1875, 16.2812, 16.875, 17.375, 17.0937, 18.0937, 19.4375]
step_baseline = [8192, 16384, 24576, 32768, 40960, 49152, 57344, 65536, 73728, 81920, 90112, 98304, 106496, 114688, 122880, 131072, 139264, 147456, 155648, 163840, 172032, 180224, 188416, 196608, 204800, 212992, 221184, 229376, 237568, 245760, 253952, 262144, 270336, 278528, 286720, 294912, 303104, 311296, 319488, 327680, 335872, 344064, 352256, 360448, 368640, 376832, 385024, 393216, 401408, 409600, 417792, 425984, 434176, 442368, 450560, 458752, 466944, 475136, 483328, 491520, 499712, 507904, 516096, 524288, 532480, 540672, 548864, 557056, 565248, 573440, 581632, 589824, 598016, 606208, 614400, 622592, 630784, 638976, 647168, 655360, 663552, 671744, 679936, 688128, 696320, 704512, 712704, 720896, 729088, 737280, 745472, 753664, 761856, 770048, 778240, 786432, 794624, 802816, 811008, 819200, 827392, 835584, 843776, 851968, 860160, 868352, 876544, 884736, 892928, 901120, 909312, 917504, 925696, 933888, 942080, 950272, 958464, 966656, 974848, 983040, 991232, 999424, 1007616, 1015808, 1024000, 1032192, 1040384, 1048576, 1056768, 1064960, 1073152, 1081344, 1089536, 1097728, 1105920, 1114112, 1122304, 1130496, 1138688, 1146880, 1155072, 1163264, 1171456, 1179648, 1187840, 1196032, 1204224, 1212416, 1220608, 1228800, 1236992, 1245184, 1253376, 1261568, 1269760, 1277952, 1286144, 1294336, 1302528, 1310720, 1318912, 1327104, 1335296, 1343488, 1351680, 1359872, 1368064, 1376256, 1384448, 1392640, 1400832, 1409024, 1417216, 1425408, 1433600, 1441792, 1449984, 1458176, 1466368, 1474560, 1482752, 1490944, 1499136, 1507328, 1515520, 1523712, 1531904, 1540096, 1548288, 1556480, 1564672, 1572864, 1581056, 1589248, 1597440, 1605632, 1613824, 1622016, 1630208, 1638400, 1646592, 1654784, 1662976, 1671168, 1679360, 1687552, 1695744, 1703936, 1712128, 1720320, 1728512, 1736704, 1744896, 1753088, 1761280, 1769472, 1777664, 1785856, 1794048, 1802240, 1810432, 1818624, 1826816, 1835008, 1843200, 1851392, 1859584, 1867776, 1875968, 1884160, 1892352, 1900544, 1908736, 1916928, 1925120, 1933312, 1941504, 1949696, 1957888, 1966080, 1974272, 1982464, 1990656, 1998848, 2007040, 2015232, 2023424, 2031616, 2039808, 2048000, 2056192, 2064384, 2072576, 2080768, 2088960, 2097152, 2105344, 2113536, 2121728, 2129920, 2138112, 2146304, 2154496, 2162688, 2170880, 2179072, 2187264, 2195456, 2203648, 2211840, 2220032, 2228224, 2236416, 2244608, 2252800, 2260992, 2269184, 2277376, 2285568, 2293760, 2301952, 2310144, 2318336, 2326528, 2334720, 2342912, 2351104, 2359296, 2367488, 2375680, 2383872, 2392064, 2400256, 2408448, 2416640, 2424832, 2433024, 2441216, 2449408, 2457600, 2465792, 2473984, 2482176, 2490368, 2498560, 2506752, 2514944, 2523136, 2531328, 2539520, 2547712, 2555904, 2564096, 2572288, 2580480, 2588672, 2596864, 2605056, 2613248, 2621440, 2629632, 2637824, 2646016, 2654208, 2662400, 2670592, 2678784, 2686976, 2695168, 2703360, 2711552, 2719744, 2727936, 2736128, 2744320, 2752512, 2760704, 2768896, 2777088, 2785280, 2793472, 2801664, 2809856, 2818048, 2826240, 2834432, 2842624, 2850816, 2859008, 2867200, 2875392, 2883584, 2891776, 2899968, 2908160, 2916352, 2924544, 2932736, 2940928, 2949120, 2957312, 2965504, 2973696, 2981888, 2990080, 2998272, 3006464, 3014656, 3022848, 3031040, 3039232, 3047424, 3055616, 3063808, 3072000, 3080192, 3088384, 3096576, 3104768, 3112960, 3121152, 3129344, 3137536, 3145728, 3153920, 3162112]

# Define temporary storage - We use this to collect transitions during each iteration.
storage = Storage(env.observation_space.shape, num_steps, num_envs)
storage_val = Storage(env_val.observation_space.shape, num_steps, num_envs)

# Save rewards for plotting purposeses.
rewards_train = []
rewards_val = []
steps = []

# Define environments 
obs = env.reset()
obs_val = env_val.reset()
step = 0
save_step = 5e3

# Define paths and directories for storing output
path = os.getcwd()
work_dir = os.path.join(path, name)
if not(os.path.isdir(work_dir)):
    try:
        os.mkdir(work_dir)
        os.chdir(work_dir)
    except:
        print("Failed to create model folder.")


# _______________________________ TRAINING _______________________________

while step < total_steps:

    # Save every half-million step
    if step > save_step:

        # Save temporary model
        torch.save(policy.state_dict, 'temp_model.pt')
        
        # Save temporary rewards
        temp_rewards = {'train_reward': [x.item() for x in rewards_train],
                'val_reward': [x.item() for x in rewards_val],
                'steps': [x for x in steps]}

        out_file = open('temp_rewards.json', "w")
        json.dump(temp_rewards, out_file, indent="")
        out_file.close()

        # Update temporary saving counter
        save_step += 5e3

    
    ############## USE POLICY ##############

    policy.eval()

    # collect data for num_steps steps
    for _ in range(num_steps):
        # Use policy
        action, log_prob, value = policy.act(obs)
        action_val, log_prob_val, value_val = policy.act(obs_val)
        
        # Take step in environment
        next_obs, reward, done, info = env.step(action)
        next_obs_val, reward_val, done_val, info_val = env_val.step(action_val) 

        # Store data
        storage.store(obs, action, reward, done, info, log_prob, value)
        storage_val.store(obs_val, action_val, reward_val, done_val, info_val, log_prob_val, value_val)

        # Update current observation
        obs = next_obs
        obs_val = next_obs_val # CHANGE BY ADDING THIS LINE

    # Add the last observation to collected data
    _, _, value = policy.act(obs)
    storage.store_last(obs, value)

    # Compute return and advantage
    storage.compute_return_advantage()


    ############## OPTIMIZE POLICY ##############

    policy.train()
    for epoch in range(num_epochs):

        # Iterate over batches of transitions
        generator = storage.get_generator(batch_size)
        for batch in generator:
            b_obs, b_action, b_log_prob, b_value, b_returns, b_advantage = batch

            # Get current policy outputs
            new_dist, new_value = policy(b_obs)
            new_log_prob = new_dist.log_prob(b_action)

            # Clipped policy objective
            ratio = torch.exp(new_log_prob - b_log_prob)
            clipped_ratio = ratio.clamp(min=1.0 - eps, max=1.0 + eps)
            policy_reward = torch.min(ratio*b_advantage, clipped_ratio*b_advantage)
            pi_loss = policy_reward.mean() 

            # Clipped value function objective
            clipped_value = b_value + (new_value - b_value).clamp(eps, eps)
            v_surr1 = (new_value - b_returns).pow(2)
            v_surr2 = (clipped_value - b_returns).pow(2)
            value_loss = 0.5 * torch.max(v_surr1, v_surr2).mean()

            # clipped_value = (new_value - b_value)**2
            # value_loss = clipped_value.mean()

            # Entropy loss
            entropy_loss = new_dist.entropy().mean()

            # Backpropagate losses
            loss = -pi_loss  + value_coef*value_loss - entropy_coef*entropy_loss # Minimize loss <=> maximize objective (notice sign opposed to paper)
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_eps)

            # Update policy
            optimizer.step()
            optimizer.zero_grad()


    ############## PLOT CURRENT PERFORMANCE ##############

    # Update stats
    step += num_envs * num_steps
    
    # Save rewards for plots.
    rewards_train.append(storage.get_reward(normalized_reward=normalize_reward))
    rewards_val.append(storage_val.get_reward(normalized_reward=normalize_reward))
    steps.append(step)

    # Plot training and validation reward vs baseline.
    clear_output(wait=True)
    plt.subplots(figsize=(10,6))
    plt.plot(step_baseline, mean_rew_baseline, 
            color="darksalmon", label="Baseline")
    plt.plot(steps, rewards_train, color="darkslategray", 
            label='Training reward: ' + str(rewards_train[-1].item()))
    plt.plot(steps, rewards_val, color="darkolivegreen", 
            label='Validation reward: ' + str(rewards_val[-1].item()))
    plt.xlabel("steps")
    plt.ylabel("Mean reward")
    plt.legend()
    plt.show()
    #print("Saving figure")
    # Save final plot.
    plt.savefig('plot_' + name + '.pdf', bbox_inches='tight')

print('Completed training!')

# Save model.
torch.save(policy.state_dict, 'model_' + name + '.pt')

# Save rewards.
temp_dict = {'train_reward': [x.item() for x in rewards_train],
        'val_reward': [x.item() for x in rewards_val],
        'steps': [x for x in steps]}

out_file = open('rewards.json', "w")
json.dump(temp_rewards, out_file, indent="")
out_file.close()


save_movie(policy, env_val)