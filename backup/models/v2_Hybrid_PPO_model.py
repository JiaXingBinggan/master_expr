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
class Critic(nn.Module):
    def __init__(self, input_dims):
        super(Critic, self).__init__()
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
            nn.Linear(neuron_nums[2], 1)
        )

    def evaluate(self, input):
        state_value = self.mlp(input)

        return state_value


class Discrete_Actor(nn.Module):
    def __init__(self, input_dims, action_dims):
        super(Discrete_Actor, self).__init__()
        self.input_dims = input_dims
        self.action_dims = action_dims

        neuron_nums = [300, 300, 300]
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
            nn.Linear(neuron_nums[2], action_dims),
            nn.Softmax(dim=-1)
        )

    def forward(self, input):
        d_values = self.mlp(input)
        d_action_dist = Categorical(d_values)
        d_actions = d_action_dist.sample()
        d_action_logprobs = d_action_dist.log_prob(d_actions)
        ensemble_d_actions = d_actions + 2

        return_d_as = (d_actions.view(-1, 1), d_action_logprobs.view(-1, 1), ensemble_d_actions.view(-1, 1))
        return return_d_as

    def evaluate(self, input, d_a):
        d_action_values = self.mlp(input)
        d_action_dist = Categorical(d_action_values)
        d_actions_logprobs = d_action_dist.log_prob(d_a.squeeze(1)).view(-1, 1)
        d_action_entropy = d_action_dist.entropy().view(-1, 1)

        return d_actions_logprobs, d_action_entropy

    def best_a(self, input):
        d_action_values = self.mlp(input)

        return d_action_values


class Continuous_Actor(nn.Module):
    def __init__(self, input_dims, action_dims):
        super(Continuous_Actor, self).__init__()
        self.input_dims = input_dims
        self.action_dims = action_dims

        neuron_nums = [300, 300, 300]
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
            nn.Linear(neuron_nums[2], action_dims),
            nn.Tanh()
        )

        self.action_std = torch.ones(1, action_dims).cuda()

    def forward(self, input):
        c_action_means = self.mlp(input)
        c_action_dist = Normal(c_action_means, self.action_std)
        c_actions = torch.clamp(c_action_dist.sample(), -1, 1)
        c_action_logprobs = c_action_dist.log_prob(c_actions)
        ensemble_c_actions = torch.softmax(c_actions, dim=-1)

        return_c_as = (c_actions, c_action_logprobs, ensemble_c_actions)

        return return_c_as

    def evaluate(self, input, c_a):
        c_action_means = self.mlp(input)
        c_action_dist = Normal(c_action_means, self.action_std)
        c_action_logprobs = c_action_dist.log_prob(c_a)
        c_action_entropy = c_action_dist.entropy()

        return c_action_logprobs, c_action_entropy

    def best_a(self, input):
        c_actions = self.mlp(input)

        return c_actions

class Hybrid_PPO_Model():
    def __init__(
            self,
            feature_nums,
            field_nums=15,
            latent_dims=5,
            action_nums=2,
            campaign_id='1458',
            init_lr_c=1e-3,
            init_lr_a=1e-4,
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
        self.init_lr_c = init_lr_c
        self.init_lr_a = init_lr_a
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

        self.critic = Critic(self.input_dims).to(self.device)
        self.d_actor = Discrete_Actor(self.input_dims, self.action_nums - 1).to(device)
        self.c_actor = Continuous_Actor(self.input_dims, self.action_nums).to(device)

        self.critic_ = Critic(self.input_dims).to(self.device)
        self.d_actor_ = Discrete_Actor(self.input_dims, self.action_nums - 1).to(device)
        self.c_actor_ = Continuous_Actor(self.input_dims, self.action_nums).to(device)

        # 优化器
        self.optimizer_c = torch.optim.Adam(self.critic.parameters(), lr=self.init_lr_c, weight_decay=1e-5)
        self.optimizer_d_a = torch.optim.Adam(self.d_actor.parameters(), lr=self.init_lr_a, weight_decay=1e-5)
        self.optimizer_c_a = torch.optim.Adam(self.c_actor.parameters(), lr=self.init_lr_a, weight_decay=1e-5)

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
        self.c_actor.eval()
        with torch.no_grad():
            c_a = self.c_actor.forward(state)
        self.c_actor.train()

        self.d_actor.eval()
        with torch.no_grad():
            d_a = self.d_actor.forward(state)
        self.d_actor.train()

        return c_a, d_a

    def choose_best_a(self, state):
        self.c_actor.eval()
        with torch.no_grad():
            c_actions = self.c_actor.best_a(state)

        self.d_actor.eval()
        with torch.no_grad():
            d_actions = self.d_actor.best_a(state)

        ensemble_c_actions = torch.softmax(c_actions, dim=-1)
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

        value_of_states_ = self.critic.evaluate(states_)  # 下一状态的V值
        value_of_states = self.critic.evaluate(states)  # 当前状态的V值

        td_target = rewards + self.gamma * value_of_states_  # 也可以采用累计折扣奖励
        deltas = td_target - value_of_states

        advantages = torch.zeros(size=[len(deltas), 1]).to(self.device)
        advantage = 0.0
        for i, deltas in enumerate(reversed(deltas)):
            advantage = self.gamma * self.lamda * advantage + deltas.item()
            advantages[i, :] = advantage

        # Normalizing the rewards
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        for _ in range(self.k_epochs):
            state_values = self.critic.evaluate(states)
            c_a_logprobs, c_a_entropy = self.c_actor.evaluate(states, old_c_a)
            d_a_logprobs, d_a_entropy = self.d_actor.evaluate(states, old_d_a)

            # Update Continuous Actor
            ratios = torch.exp(c_a_logprobs - old_c_a_logprobs)
            c_a_surr1 = ratios * advantages
            c_a_surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            c_a_loss = -torch.min(c_a_surr1, c_a_surr2).mean()
            # c_a_entropy_loss = 0.01 * c_a_entropy.mean()

            self.optimizer_c_a.zero_grad()
            c_a_loss.backward()
            self.optimizer_c_a.step()

            # Update Discrete Actor
            ratios = torch.exp(d_a_logprobs - old_d_a_logprobs)
            d_a_surr1 = ratios * advantages
            d_a_surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            d_a_loss = -torch.min(d_a_surr1, d_a_surr2).mean()
            # d_a_entropy_loss = 0.01 * d_a_entropy.mean()

            self.optimizer_d_a.zero_grad()
            d_a_loss.backward()
            self.optimizer_d_a.step()

            # Update Value Layer(Critic)
            critic_loss = self.loss_func(state_values, td_target.detach())
            self.optimizer_c.zero_grad()
            critic_loss.backward()
            self.optimizer_c.step()

            # loss = c_a_loss - c_a_entropy_loss + d_a_loss - d_a_entropy_loss + 0.5 * critic_loss

            # print('3', datetime.datetime.now())

            # print(self.hybrid_actor_critic.Critic.weight)
            # take gradient step
            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()
            # print(self.hybrid_actor_critic.Critic.weight)
            # print('4', datetime.datetime.now())
            # print("第个epoch的学习率：%f" % (self.optimizer.param_groups[0]['lr']))


            # return_loss = loss.mean().item()
        # print('5', datetime.datetime.now())

        self.critic_.load_state_dict(self.critic.state_dict())
        self.d_actor_.load_state_dict(self.d_actor.state_dict())
        self.c_actor_.load_state_dict(self.c_actor.state_dict())

        return critic_loss.item()


