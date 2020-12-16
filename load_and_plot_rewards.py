
def plot_experiments(name, comparison_name):
    path = os.getcwd()
    compare_dir = os.path.join(path, comparison_name)
    print("PATH IS: ", path, compare_dir)
    if not(os.path.isdir(compare_dir)):
        except("Failed to find folder for comparison.")
    else:
        with open(compare_dir + "/temp_rewards.json") as json_file:
            data = json.load(json_file)
            mean_rew_baseline = data['val_reward']
            step_baseline = data['steps']

    # Plot training and validation reward vs baseline.
    clear_output(wait=True)
    plt.subplots(figsize=(10,6))
    plt.plot(step_baseline, mean_rew_baseline, 
            color="darksalmon", label=comparison_name)
    plt.plot(steps, rewards_train, color="darkslategray", 
            label='Training reward: ' + str(rewards_train[-1].item()))
    plt.plot(steps, rewards_val, color="darkolivegreen", 
            label='Validation reward: ' + str(rewards_val[-1].item()))
    plt.xlabel("steps")
    plt.ylabel("Mean reward")
    plt.legend()
    #plt.show()
    #print("Saving figure")
    # Save final plot.
    plt.savefig('plot_' + name + '.pdf', bbox_inches='tight')
    plt.close()



name = '9_model_5_coinrun_03e8_steps'
comparison_name = '9_model_5_coinrun'

experiments = [name, comparison_name]

plot_experiments(name, comparison_name)