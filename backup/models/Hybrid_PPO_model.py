import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from torch.distributions import Normal, Categorical
import datetime

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(1)
class Hybrid_Actor_Critic(nn.Module):
    def __init__(self, input_dims, action_nums):
        super(Hybrid_Actor_Critic, self).__init__()
        self.input_dims = input_dims

        neuron_nums = [300, 300, 300]

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dims, neuron_nums[0]),
            nn.BatchNorm1d(neuron_nums[0]),
            nn.ReLU(),
            nn.Linear(neuron_nums[0], neuron_nums[1]),
            nn.BatchNorm1d(neuron_nums[1]),
            nn.ReLU(),
            nn.Linear(neuron_nums[1], neuron_nums[2]),
            nn.BatchNorm1d(neuron_nums[2]),
            nn.ReLU(),
        )

        # Critic
        self.Critic = nn.Linear(neuron_nums[2], 1)

        # Continuous_Actor
        self.Continuous_Actor = nn.Linear(neuron_nums[2], action_nums)

        # Discrete_Actor
        self.Discrete_Actor = nn.Linear(neuron_nums[2], action_nums - 1)

        self.action_std = torch.ones(1, action_nums).cuda()
        self.action_nums = action_nums

    def act(self, input):
        mlp_out = self.mlp(input)

        c_action_means = torch.softmax(self.Continuous_Actor(mlp_out), dim=-1)
        c_action_dist = Normal(c_action_means, self.action_std)
        c_actions = c_action_dist.sample()
        c_action_logprobs = c_action_dist.log_prob(c_actions)
        ensemble_c_actions = torch.softmax(c_actions, dim=-1)

        return_c_as = (c_actions, c_action_logprobs, ensemble_c_actions)

        d_actions = torch.softmax(self.Discrete_Actor(mlp_out), dim=-1)
        d_action_dist = Categorical(d_actions)
        d_actions = d_action_dist.sample()
        d_action_logprobs = d_action_dist.log_prob(d_actions)
        ensemble_d_actions = d_actions + 2

        return_d_as = (d_actions.view(-1, 1), d_action_logprobs.view(-1, 1), ensemble_d_actions.view(-1, 1))

        return return_c_as, return_d_as

    def best_a(self, input):
        mlp_out = self.mlp(input)

        c_actions = torch.softmax(self.Continuous_Actor(mlp_out), dim=-1)
        d_actions = torch.softmax(self.Discrete_Actor(mlp_out), dim=-1)

        return c_actions, d_actions

    def evaluate(self, input, c_a, d_a):
        mlp_out = self.mlp(input)

        state_value = self.Critic(mlp_out)

        c_action_means = torch.softmax(self.Continuous_Actor(mlp_out), dim=-1)
        c_action_dist = Normal(c_action_means, self.action_std)
        c_action_logprobs = c_action_dist.log_prob(c_a)
        c_action_entropy = c_action_dist.entropy()

        d_action_values = self.Discrete_Actor(mlp_out)
        d_action_dist = Categorical(d_action_values)
        d_actions_logprobs = d_action_dist.log_prob(d_a.squeeze(1)).view(-1, 1)
        d_action_entropy = d_action_dist.entropy().view(-1, 1)

        return state_value, c_action_logprobs, c_action_entropy, d_actions_logprobs, d_action_entropy

class Hybrid_PPO_Model():
    def __init__(
            self,
            feature_nums,
            field_nums=15,
            latent_dims=5,
            action_nums=2,
            campaign_id='1458',
            init_lr=1e-2,
            train_epochs=500,
            reward_decay=1,
            lr_lamda=0.01,
            memory_size=4096000, # 设置为DataLoader的batch_size * n
            batch_size=256,
            tau=0.005,  # for target network soft update
            k_epochs=3, # update policy for k epochs
            eps_clip=0.2, # clip parameter for ppo
            device='cuda:0',
    ):
        self.feature_nums = feature_nums
        self.field_nums = field_nums
        self.action_nums = action_nums
        self.campaign_id = campaign_id
        self.init_lr = init_lr
        self.train_epochs = train_epochs
        self.gamma = reward_decay
        self.latent_dims = latent_dims
        self.lr_lamda = lr_lamda
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.tau = tau
        self.device = device
        self.action_std = 0.5
        self.k_epochs = k_epochs
        self.eps_clip = eps_clip
        self.lamda = 0.95 # GAE泛化估计

        self.memory_counter = 0

        self.input_dims = self.field_nums * (self.field_nums - 1) // 2 + self.field_nums * self.latent_dims

        self.memory_state = torch.zeros(size=[self.memory_size, self.field_nums]).to(self.device)
        self.memory_c_a = torch.zeros(size=[self.memory_size, self.action_nums]).to(self.device)
        self.memory_c_logprobs = torch.zeros(size=[self.memory_size, self.action_nums]).to(self.device)
        self.memory_d_a = torch.zeros(size=[self.memory_size, 1]).to(self.device)
        self.memory_d_logprobs = torch.zeros(size=[self.memory_size, 1]).to(self.device)
        self.memory_reward = torch.zeros(size=[self.memory_size, 1]).to(self.device)

        self.hybrid_actor_critic = Hybrid_Actor_Critic(self.input_dims, self.action_nums).to(self.device)

        self.hybrid_actor_critic_old = Hybrid_Actor_Critic(self.input_dims, self.action_nums).to(self.device)

        # 优化器
        self.optimizer = torch.optim.Adam(self.hybrid_actor_critic.parameters(), lr=self.init_lr, weight_decay=1e-5)

        self.loss_func = nn.MSELoss()

    def store_memory(self, states, c_a, c_logprobs, d_a, d_logprobs, rewards):
        transition_lens = len(states)

        # 由于已经定义了经验池的memory_size，如果超过此大小，旧的memory则被新的memory替换
        index_start = self.memory_counter % self.memory_size
        # index_end = (self.memory_counter + transition_lens) % self.memory_size
        index_end = self.memory_counter + transition_lens

        self.memory_state[index_start: index_end, :] = states
        self.memory_c_a[index_start: index_end, :] = c_a
        self.memory_c_logprobs[index_start: index_end, :] = c_logprobs
        self.memory_d_a[index_start: index_end, :] = d_a
        self.memory_d_logprobs[index_start: index_end, :] = d_logprobs
        self.memory_reward[index_start: index_end, :] = rewards

        # self.memory_counter += transition_lens

    def choose_a(self, state):
        self.hybrid_actor_critic.eval()
        with torch.no_grad():
            c_a, d_a = self.hybrid_actor_critic.act(state)

        return c_a, d_a

    def choose_best_a(self, state):
        self.hybrid_actor_critic.eval()
        with torch.no_grad():
            ensemble_c_actions, d_actions = self.hybrid_actor_critic.best_a(state)
            ensemble_d_actions = torch.argsort(-d_actions)[:, 0] + 2

        return ensemble_c_actions, ensemble_d_actions.view(-1, 1)

    def memory(self):
        states = self.memory_state.long()
        states_ = self.memory_state.long()
        old_c_a = self.memory_c_a
        old_c_a_logprobs = self.memory_c_logprobs
        old_d_a = self.memory_d_a
        old_d_a_logprobs = self.memory_d_logprobs
        rewards = self.memory_reward

        return states, states_, old_c_a, old_c_a_logprobs, old_d_a, old_d_a_logprobs, rewards

    def learn(self, states, states_, old_c_a, old_c_a_logprobs, old_d_a, old_d_a_logprobs, rewards):
        return_loss = 0
        # print('1', datetime.datetime.now())

        value_of_states_ = self.hybrid_actor_critic.evaluate(states_, old_c_a, old_d_a)  # 下一状态的V值
        value_of_states = self.hybrid_actor_critic.evaluate(states, old_c_a, old_d_a)  # 当前状态的V值

        td_target = rewards + self.gamma * value_of_states_[0]  # 也可以采用累计折扣奖励
        deltas = td_target - value_of_states[0]

        advantages = torch.zeros(size=[len(deltas), 1]).to(self.device)
        advantage = 0.0
        for i, deltas in enumerate(reversed(deltas)):
            advantage = self.gamma * self.lamda * advantage + deltas.item()
            advantages[i, :] = advantage

        # Normalizing the rewards
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        # print('2', datetime.datetime.now())

        for _ in range(self.k_epochs):
            state_values, c_a_logprobs, c_a_entropy, d_a_logprobs, d_a_entropy = self.hybrid_actor_critic.evaluate(states, old_c_a, old_d_a)

            # Update Continuous Actor
            ratios = torch.exp(c_a_logprobs - old_c_a_logprobs)
            c_a_surr1 = ratios * advantages
            c_a_surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            c_a_loss = -torch.min(c_a_surr1, c_a_surr2).mean()
            c_a_entropy_loss = 0.01 * c_a_entropy.mean()

            # Update Discrete Actor
            ratios = torch.exp(d_a_logprobs - old_d_a_logprobs)
            d_a_surr1 = ratios * advantages
            d_a_surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            d_a_loss = -torch.min(d_a_surr1, d_a_surr2).mean()
            d_a_entropy_loss = 0.01 * d_a_entropy.mean()

            # Update Value Layer(Critic)
            critic_loss = self.loss_func(state_values, td_target.detach())

            loss = c_a_loss - c_a_entropy_loss + d_a_loss - d_a_entropy_loss + 0.5 * critic_loss

            # print('3', datetime.datetime.now())

            # print(self.hybrid_actor_critic.Critic.weight)
            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # print(self.hybrid_actor_critic.Critic.weight)
            # print('4', datetime.datetime.now())
            # print("第个epoch的学习率：%f" % (self.optimizer.param_groups[0]['lr']))


            return_loss = loss.mean().item()
        # print('5', datetime.datetime.now())

        self.hybrid_actor_critic_old.load_state_dict(self.hybrid_actor_critic.state_dict())

        return return_loss


