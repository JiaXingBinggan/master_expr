import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.distributions import MultivariateNormal, Categorical

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(1)

class Discrete_Actor(nn.Module):
    def __init__(self, input_dims, action_nums):
        super(Discrete_Actor, self).__init__()
        self.input_dims = input_dims

        deep_input_dims = self.input_dims

        self.bn_input = nn.BatchNorm1d(self.input_dims)

        neuron_nums = [300, 300, 300]
        self.mlp = nn.Sequential(
            nn.Linear(deep_input_dims, neuron_nums[0]),
            nn.BatchNorm1d(neuron_nums[0]),
            nn.ReLU(),
            nn.Linear(neuron_nums[0], neuron_nums[1]),
            nn.BatchNorm1d(neuron_nums[1]),
            nn.ReLU(),
            nn.Linear(neuron_nums[1], neuron_nums[2]),
            nn.BatchNorm1d(neuron_nums[2]),
            nn.ReLU(),
            nn.Linear(neuron_nums[2], action_nums)
        )

    def forward(self, input):
        q_values = self.mlp(self.bn_input(input))

        return q_values

class Continuous_Actor(nn.Module):
    def __init__(self, input_dims, action_nums):
        super(Continuous_Actor, self).__init__()
        self.input_dims = input_dims

        self.bn_input = nn.BatchNorm1d(self.input_dims + 1)

        deep_input_dims = self.input_dims + 1
        neuron_nums = [300, 300, 300]

        self.mlp = nn.Sequential(
            nn.Linear(deep_input_dims, neuron_nums[0]),
            nn.BatchNorm1d(neuron_nums[0]),
            nn.ReLU(),
            nn.Linear(neuron_nums[0], neuron_nums[1]),
            nn.BatchNorm1d(neuron_nums[1]),
            nn.ReLU(),
            nn.Linear(neuron_nums[1], neuron_nums[2]),
            nn.BatchNorm1d(neuron_nums[2]),
            nn.ReLU(),
            nn.Linear(neuron_nums[2], action_nums),
            nn.Tanh(),
        )

    def forward(self, input, discrete_a):
        obs = self.bn_input(torch.cat([input, discrete_a], dim=1))
        out = self.mlp(obs)

        return out


class Critic(nn.Module):
    def __init__(self, input_dims, c_action_nums):
        super(Critic, self).__init__()
        self.input_dims = input_dims
        self.c_action_nums = c_action_nums

        self.bn_input = nn.BatchNorm1d(self.input_dims + 1)

        deep_input_dims = self.input_dims + self.c_action_nums + 1

        neuron_nums = [300, 300, 300]
        self.mlp = nn.Sequential(
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

    def forward(self, input, action, discrete_a):
        obs = self.bn_input(torch.cat([input, discrete_a], dim=1))

        cat = torch.cat([obs, action], dim=1)

        q_values = self.mlp(cat)

        return q_values

class Hybrid_RL_Model():
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
            tau=0.005, # for target network soft update
            device='cuda:0',
    ):
        self.feature_nums = feature_nums
        self.field_nums = field_nums
        self.c_a_action_nums = action_nums
        self.d_actions_nums = action_nums - 1
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

        self.memory_state = torch.zeros(size=[self.memory_size, self.field_nums]).to(self.device)
        self.memory_action_reward = torch.zeros(size=[self.memory_size, self.c_a_action_nums + 1]).to(self.device)
        self.memory_discrete_action = torch.zeros(size=[self.memory_size, 1]).to(self.device)

        self.Continuous_Actor = Continuous_Actor(self.input_dims, self.c_a_action_nums).to(self.device)
        self.Discrete_Actor = Discrete_Actor(self.input_dims, self.d_actions_nums).to(self.device)
        self.Critic = Critic(self.input_dims, self.c_a_action_nums).to(self.device)
        
        self.Continuous_Actor_ = Continuous_Actor(self.input_dims, self.c_a_action_nums).to(self.device)
        self.Discrete_Actor_ = Discrete_Actor(self.input_dims, self.d_actions_nums).to(self.device)
        self.Critic_ = Critic(self.input_dims, self.c_a_action_nums).to(self.device)

        # 优化器
        self.optimizer_c_a = torch.optim.Adam(self.Continuous_Actor.parameters(), lr=self.lr_C_A, weight_decay=1e-5)
        self.optimizer_d_a = torch.optim.Adam(self.Discrete_Actor.parameters(), lr=self.lr_D_A, weight_decay=1e-5)
        self.optimizer_c = torch.optim.Adam(self.Critic.parameters(), lr=self.lr_C, weight_decay=1e-5)

        self.loss_func = nn.MSELoss(reduction='mean')

    def store_transition(self, features, action_rewards, discrete_actions):
        transition_lens = len(features)

        # 由于已经定义了经验池的memory_size，如果超过此大小，旧的memory则被新的memory替换
        index_start = self.memory_counter % self.memory_size
        index_end = (self.memory_counter + transition_lens) % self.memory_size

        if index_end > index_start:
            self.memory_state[index_start: index_end, :] = features  # 替换
            self.memory_action_reward[index_start: index_end, :] = action_rewards
            self.memory_discrete_action[index_start: index_end, :] = discrete_actions
        else:
            replace_len_1 = self.memory_size - index_start
            self.memory_state[index_start: self.memory_size, :] = features[0: replace_len_1]
            self.memory_action_reward[index_start: self.memory_size, :] = action_rewards[0: replace_len_1]
            self.memory_discrete_action[index_start: self.memory_size, :] = discrete_actions[0: replace_len_1]

            replace_len_2 = transition_lens - replace_len_1
            self.memory_state[0: replace_len_2, :] = features[replace_len_1: transition_lens]
            self.memory_action_reward[0: replace_len_2, :] = action_rewards[replace_len_1: transition_lens]
            self.memory_discrete_action[0: replace_len_2, :] = discrete_actions[replace_len_1: transition_lens]

        self.memory_counter += transition_lens

    def choose_continuous_action(self, state, discrete_a, exploration_rate):
        self.Continuous_Actor.eval()
        with torch.no_grad():
            action_mean = self.Continuous_Actor.forward(state, discrete_a)
        # random_action = torch.normal(action_mean, exploration_rate)
        random_action = torch.clamp(torch.normal(action_mean, exploration_rate), -1, 1)

        ensemble_c_actions = torch.softmax(random_action, dim=-1)  # 模型所需的动作

        self.Continuous_Actor.train()

        return random_action, ensemble_c_actions

    def choose_discrete_action(self, state, exploration_rate):
        self.Discrete_Actor.eval()
        with torch.no_grad():
            action_values = self.Discrete_Actor.forward(state)
        random_values = torch.normal(action_values, exploration_rate)
        ensemble_d_actions = torch.argsort(-random_values)[:, 0] + 2
        self.Discrete_Actor.train()

        return ensemble_d_actions.view(-1, 1)

    def choose_best_continuous_action(self, state, discrete_a):
        self.Continuous_Actor.eval()
        with torch.no_grad():
            action = torch.softmax(self.Continuous_Actor.forward(state, discrete_a), dim=-1)

        return action
    
    def choose_best_discrete_action(self, state):
        self.Discrete_Actor.eval()
        with torch.no_grad():
            action_values = self.Discrete_Actor.forward(state)
            action = torch.argsort(-action_values)[:, 0] + 2

        return action.view(-1, 1)

    def soft_update(self, net, net_target):
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def learn(self, embedding_layer):
        # sample
        if self.memory_counter > self.memory_size:
            sample_index = torch.LongTensor(random.sample(range(self.memory_size), self.batch_size)).to(self.device)
        else:
            sample_index = torch.LongTensor(random.sample(range(self.memory_counter), self.batch_size)).to(self.device)

        batch_memory_states = self.memory_state[sample_index, :].long()
        batch_memory_action_rewards = self.memory_action_reward[sample_index, :]
        b_discrete_a = self.memory_discrete_action[sample_index, :]

        b_s = embedding_layer.forward(batch_memory_states)
        b_a = batch_memory_action_rewards[:, 0: self.c_a_action_nums]
        b_r = torch.unsqueeze(batch_memory_action_rewards[:, self.c_a_action_nums], 1)
        b_s_ = b_s # embedding_layer.forward(batch_memory_states)

        # D_A
        q_eval = self.Discrete_Actor.forward(b_s).gather(1,
                                                         b_discrete_a.long() - 2)  # shape (batch,1), gather函数将对应action的Q值提取出来做Bellman公式迭代
        q_next = self.Discrete_Actor_.forward(b_s_)

        # # # 下一状态s的eval_net值
        # q_eval_next = self.Discrete_Actor.forward(b_s_)
        # max_b_a_next = torch.unsqueeze(torch.max(q_eval_next, 1)[1], 1)  # 选择最大Q的动作
        # select_q_next = q_next.gather(1, max_b_a_next)

        q_target = b_r + self.gamma * q_next.max(1)[0].view(-1, 1)  # shape (batch, 1)

        # 训练eval_net
        d_a_loss = self.loss_func(q_eval, q_target.detach())

        self.optimizer_d_a.zero_grad()
        d_a_loss.backward()
        self.optimizer_d_a.step()

        d_a_loss_r = d_a_loss.item()

        # Critic
        # evaluate_discrete_action = (torch.argsort(-self.Discrete_Actor_.forward(b_s_))[:, 0] + 2).view(-1, 1).float()
        q_target = b_r + self.gamma * self.Critic_.forward(b_s_, self.Continuous_Actor_.forward(b_s_,
                                                           b_discrete_a),
                                                           b_discrete_a)
        q = self.Critic.forward(b_s, b_a, b_discrete_a)

        td_error = self.loss_func(q, q_target.detach())

        self.optimizer_c.zero_grad()
        td_error.backward()
        self.optimizer_c.step()

        td_error_r = td_error.item()

        # C_A
        c_a_loss = -self.Critic.forward(b_s, self.Continuous_Actor.forward(b_s, b_discrete_a), b_discrete_a).mean()

        self.optimizer_c_a.zero_grad()
        c_a_loss.backward()
        self.optimizer_c_a.step()
        c_a_loss_r = c_a_loss.item()

        return td_error_r, c_a_loss_r, d_a_loss_r


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
