import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.distributions import MultivariateNormal, Categorical
import datetime
from torch.distributions import Normal, Categorical, MultivariateNormal

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(1)

class Memory(object):
    def __init__(self, memory_size, transition_lens, device):
        self.device = device
        self.transition_lens = transition_lens # 存储的数据长度

        self.memory_size = memory_size
        self.memory_counter = 0
        self.memory = torch.zeros(size=[memory_size, transition_lens]).to(self.device)

    def add(self, transitions): # td_error是tensor矩阵
        transition_lens = len(transitions)

        memory_start = self.memory_counter % self.memory_size
        memory_end = (self.memory_counter + len(transitions)) % self.memory_size

        if memory_end > memory_start:
            self.memory[memory_start: memory_end, :] = transitions
        else:
            replace_len_1 = self.memory_size - memory_start
            self.memory[memory_start: self.memory_size, :] = transitions[0: replace_len_1]

            replace_len_2 = transition_lens - replace_len_1
            self.memory[:replace_len_2, :] = transitions[replace_len_1: transition_lens]

        self.memory_counter += len(transitions)

    def stochastic_sample(self, batch_size):
        if self.memory_counter >= self.memory_size:
            sample_indexs = torch.Tensor(random.sample(range(self.memory_size), batch_size)).long().to(self.device)
        else:
            sample_indexs = torch.Tensor(random.sample(range(self.memory_counter), batch_size)).long().to(self.device)

        batch = self.memory[sample_indexs]

        return batch

class Critic(nn.Module):
    def __init__(self, input_dims, c_action_nums, d_action_nums):
        super(Critic, self).__init__()
        self.input_dims = input_dims
        self.c_action_nums = c_action_nums
        self.d_action_nums = d_action_nums

        self.bn_input = nn.BatchNorm1d(self.input_dims)

        deep_input_dims = self.input_dims + self.c_action_nums + self.d_action_nums

        neuron_nums = [512, 256, 512]
        self.mlp_1 = nn.Sequential(
            nn.Linear(deep_input_dims, neuron_nums[0]),
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

        # self.mlp_1[9].weight.data.uniform_(-3e-3, 3e-3)
        # self.mlp_1[9].bias.data.uniform_(-3e-3, 3e-3)

        self.mlp_2 = nn.Sequential(
            nn.Linear(deep_input_dims, neuron_nums[0]),
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

        # self.mlp_2[9].weight.data.uniform_(-3e-3, 3e-3)
        # self.mlp_2[9].bias.data.uniform_(-3e-3, 3e-3)

    def evaluate(self, input, c_actions, d_actions):
        obs = self.bn_input(input)
        # obs = input
        cat = torch.cat([obs, c_actions, d_actions], dim=1)

        q_out_1 = self.mlp_1(cat)

        q_out_2 = self.mlp_2(cat)

        return q_out_1, q_out_2

    def evaluate_q_1(self, input, c_actions, d_actions):
        obs = self.bn_input(input)
        # obs = input
        cat = torch.cat([obs, c_actions, d_actions], dim=1)

        q_out_1 = self.mlp_1(cat)

        return q_out_1

class hybrid_actors(nn.Module):
    def __init__(self, input_dims, action_nums):
        super(hybrid_actors, self).__init__()
        self.input_dims = input_dims
        self.c_action_dims = action_nums
        self.d_action_dims = action_nums

        self.bn_input = nn.BatchNorm1d(self.input_dims)

        neuron_nums = [512, 256, 512]
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dims, neuron_nums[0]),
            nn.BatchNorm1d(neuron_nums[0]),
            nn.ReLU(),
            nn.Linear(neuron_nums[0], neuron_nums[1]),
            nn.BatchNorm1d(neuron_nums[1]),
            nn.ReLU(),
            nn.Linear(neuron_nums[1], neuron_nums[2]),
            nn.BatchNorm1d(neuron_nums[2]),
            nn.ReLU()
        )# 特征提取层

        self.c_action_layer = nn.Sequential(
            nn.Linear(neuron_nums[2], self.c_action_dims),
            nn.Tanh()
        )

        # self.c_action_layer[0].weight.data.uniform_(-3e-3, 3e-3)
        # self.c_action_layer[0].bias.data.uniform_(-3e-3, 3e-3)

        self.d_action_layer = nn.Sequential(
            nn.Linear(neuron_nums[2], self.d_action_dims),
            nn.Tanh()
        )

        # self.d_action_layer[0].weight.data.uniform_(-3e-3, 3e-3)
        # self.d_action_layer[0].bias.data.uniform_(-3e-3, 3e-3)

        self.std = torch.ones(size=[1, self.c_action_dims]).cuda()
        self.mean = torch.zeros(size=[1, self.c_action_dims]).cuda()

    def act(self, input):
        obs = self.bn_input(input)
        # obs = input
        mlp_out = self.mlp(obs)

        c_action_means = self.c_action_layer(mlp_out)
        d_action_q_values = self.d_action_layer(mlp_out)

        c_actions = torch.clamp(c_action_means + torch.randn_like(c_action_means) * 0.1, -1, 1)  # 用于返回训练

        ensemble_c_actions = torch.softmax(c_actions, dim=-1)

        d_action = torch.clamp(d_action_q_values + torch.randn_like(d_action_q_values) * 0.1, -1, 1)
        ensemble_d_actions = torch.argsort(-d_action)[:, 0] + 1

        return c_actions, ensemble_c_actions, d_action, ensemble_d_actions.view(-1, 1)

    def evaluate(self, input):
        obs = self.bn_input(input)
        # obs = input
        mlp_out = self.mlp(obs)

        c_actions_means = self.c_action_layer(mlp_out)
        d_actions_q_values = self.d_action_layer(mlp_out)

        return c_actions_means, d_actions_q_values


class Hybrid_TD3_Model():
    def __init__(
            self,
            feature_nums,
            field_nums=15,
            latent_dims=5,
            action_nums=2,
            campaign_id='1458',
            lr_C_A=1e-3,
            lr_D_A=1e-3,
            lr_C=1e-2,
            reward_decay=1,
            memory_size=4096000,
            batch_size=256,
            tau=0.005,  # for target network soft update
            device='cuda:0',
    ):
        self.feature_nums = feature_nums
        self.field_nums = field_nums
        self.c_action_nums = action_nums
        self.d_action_nums = action_nums
        self.campaign_id = campaign_id
        self.lr_C_A = lr_C_A
        self.lr_D_A = lr_D_A
        self.lr_C = lr_C
        self.gamma = reward_decay
        self.latent_dims = latent_dims
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.tau = tau
        self.device = device

        self.memory_counter = 0

        self.input_dims = self.field_nums * (self.field_nums - 1) // 2 + self.field_nums * self.latent_dims

        self.memory = Memory(self.memory_size, self.field_nums + self.c_action_nums + self.d_action_nums + 2, self.device)

        self.Hybrid_Actor = hybrid_actors(self.input_dims, self.c_action_nums).to(self.device)
        self.Critic = Critic(self.input_dims, self.c_action_nums, self.d_action_nums).to(self.device)

        self.Hybrid_Actor_ = hybrid_actors(self.input_dims, self.c_action_nums).to(self.device)
        self.Critic_ = Critic(self.input_dims, self.c_action_nums,  self.d_action_nums).to(self.device)

        # 优化器
        self.optimizer_a = torch.optim.Adam(self.Hybrid_Actor.parameters(), lr=self.lr_C_A)
        self.optimizer_c = torch.optim.Adam(self.Critic.parameters(), lr=self.lr_C)

        self.loss_func = nn.MSELoss(reduction='mean')

        self.action_mean = torch.zeros(size=[1, self.c_action_nums]).to(self.device)
        self.action_std = torch.ones(size=[1, self.c_action_nums]).to(self.device)

        self.learn_iter = 0
        self.policy_freq = 4

    def store_transition(self, transitions): # 所有的值都应该弄成float
        self.memory.add(transitions)

    def choose_action(self, state):
        self.Hybrid_Actor.eval()
        with torch.no_grad():
            c_actions, ensemble_c_actions, d_q_values, ensemble_d_actions = self.Hybrid_Actor.act(state)

        self.Hybrid_Actor.train()
        return c_actions, ensemble_c_actions, d_q_values, ensemble_d_actions

    def choose_best_action(self, state):
        self.Hybrid_Actor.eval()
        with torch.no_grad():
            c_action_means, d_q_values = self.Hybrid_Actor.evaluate(state)

        ensemble_c_actions = torch.softmax(c_action_means, dim=-1)
        ensemble_d_actions = torch.argsort(-d_q_values)[:, 0] + 1

        return ensemble_d_actions.view(-1, 1), ensemble_c_actions

    def soft_update(self, net, net_target):
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def learn(self, embedding_layer):
        self.learn_iter += 1

        # sample
        batch_memory = self.memory.stochastic_sample(self.batch_size)

        b_s = embedding_layer.forward(batch_memory[:, :self.field_nums].long())
        b_c_a = batch_memory[:, self.field_nums: self.field_nums + self.c_action_nums]
        b_d_a = batch_memory[:,
                self.field_nums + self.c_action_nums: self.field_nums + self.c_action_nums + self.d_action_nums]
        b_discrete_a = torch.unsqueeze(batch_memory[:, self.field_nums + self.c_action_nums] + self.d_action_nums, 1)
        b_r = torch.unsqueeze(batch_memory[:, -1], 1)
        b_s_ = b_s  # embedding_layer.forward(batch_memory_states)

        with torch.no_grad():
            c_actions_means_next, d_actions_q_values_next = self.Hybrid_Actor_.evaluate(b_s_)

            next_c_actions = torch.clamp(c_actions_means_next + torch.clamp(torch.randn_like(c_actions_means_next) * 0.2, -0.5, 0.5), -1, 1)
            next_d_actions = torch.clamp(d_actions_q_values_next + torch.clamp(torch.randn_like(d_actions_q_values_next) * 0.2, -0.5, 0.5), -1, 1)

            q1_target, q2_target = self.Critic_.evaluate(b_s_, next_c_actions, next_d_actions)
            q_target = torch.min(q1_target, q2_target)
            q_target = b_r + self.gamma * q_target

        q1, q2 = self.Critic.evaluate(b_s, b_c_a, b_d_a)

        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.optimizer_c.zero_grad()
        critic_loss.backward()
        self.optimizer_c.step()

        critic_loss_r = critic_loss.item()

        if self.learn_iter % self.policy_freq == 0:
            c_actions_means, d_actions_q_values = self.Hybrid_Actor.evaluate(b_s)
            # Hybrid_Actor
            # c a
            c_a_loss = -self.Critic.evaluate_q_1(b_s, c_actions_means, d_actions_q_values).mean()

            # actor_loss = c_a_loss - c_actions_entropy.mean() - d_actions_entropy.mean()
            actor_loss = c_a_loss

            self.optimizer_a.zero_grad()
            actor_loss.backward()
            self.optimizer_a.step()

            self.soft_update(self.Critic, self.Critic_)
            self.soft_update(self.Hybrid_Actor, self.Hybrid_Actor_)

        return critic_loss_r

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.15, 0.01, 0.2
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
