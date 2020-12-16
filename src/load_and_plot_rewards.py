import json 
import matplotlib.pyplot as plt

def plot_experiments(experiments, colors, title):
#     steps = []
#     rewards_vals = []

    plt.subplots(figsize=(10,6))
    for (path, name), color in zip(experiments, colors):
        #path = os.getcwd()
        #compare_dir = os.path.join(path, exp)

        with open("experiments/" + path + "/rewards.json") as json_file:
            data = json.load(json_file)
            rewards_val = data['val_reward']
            steps = data['steps']

        # Plot training and validation reward vs baseline.
        plt.plot(steps, rewards_val, label=name, color=color)

    locs, labels = plt.xticks()

    #labels = [label.() + "M" for label in labels]
    labels = [str(i) + "M" for i in range(0,len(labels))]
    plt.xticks(locs[1:len(locs)-1], labels)
    #plt.ylim((0, 20))
    plt.title(title)
    plt.xlabel("Steps [millions]")
    plt.ylabel("Mean Reward")
    plt.legend()
    plt.show()



e0 = ('0_baseline', "Nature DQN (10 lvls - baseline)")
e1 = ('1_100_lvls', 'Nature DQN (100 lvls)')
e2 = ('2_500_lvls', "Nature DQN (500 lvls)")
e3 = ('3_100_lvls_impala', "Impala (100 lvls)")
e4 = ('4_500_lvls_impala', "Impala (500 lvls)")
e5 = ('5_500_lvls_impala_valclip', "Impala (val-clip)")
e6 = ('6_model_2_boss_fight', "Nature DQN")
e7 = ('7_model_5_boss_fight', "Impala")
e8 = ('8_model_2_coinrun', "DQN")
e9 = ('9_model_5_coinrun', "Impala (val-clip)")
e10 = ('10_model_2_bigfish', "Nature DQN")
e11 = ('11_model_5_bigfish', "Impala")



experiments = [e2, e4, e5]
colors = ["darkslategray", "darksalmon", "lightskyblue", "royalblue"][:len(experiments)]

plot_experiments(experiments, colors, "Star Pilot")

