import pandas as pd

reward_fi = open('reward_1_0.txt', 'r')

timesteps = []
aucs = []
rewards = []
for line in reward_fi:
    split_1 = line.strip().split('timesteps')
    if len(split_1) <= 1:
        continue
    else:
        split = line.strip().split('timesteps')[1]

        timestep = split.split('test_auc')[0]

        split_2 = split.split('test_auc')[1]
        auc = split_2.split('test_rewards')[0]
        reward = split_2.split('test_rewards')[1]

        timesteps.append(timestep)
        aucs.append(auc)
        rewards.append(reward)

reward_dict = {'timesteps': timesteps, 'test_aucs': aucs, 'test_rewards': rewards}
reward_df = pd.DataFrame(data=reward_dict, index=None)
reward_df.to_csv('reward_positive_1_negative_0.csv')