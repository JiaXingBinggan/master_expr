import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import defaultdict


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def weight_init(layers):
    # source: The other layers were initialized from uniform distributions
    # [− 1/sqrt(f) , 1/sqrt(f) ] where f is the fan-in of the layer
    for layer in layers:
        if isinstance(layer, nn.BatchNorm1d):
            layer.weight.data.fill_(1)
            layer.bias.data.zero_()
        elif isinstance(layer, nn.Linear):
            fan_in = layer.weight.data.size()[0]
            lim = 1. / np.sqrt(fan_in)
            layer.weight.data.uniform_(-0.003, 0.003)
            layer.bias.data.fill_(0)

class Net(nn.Module):
    def __init__(self, state_dims, action_dims, reward_dims, neuron_dims):
        super(Net, self).__init__()

        self.input_dims = state_dims + action_dims
        self.reward_dims = reward_dims
        self.neuron_dims = neuron_dims

        deep_input_dims = self.input_dims
        self.bn_input = nn.BatchNorm1d(deep_input_dims)
        self.bn_input.weight.data.fill_(1)
        self.bn_input.bias.data.fill_(0)

        self.layers = list()
        for neuron_num in self.neuron_dims:
            self.layers.append(nn.Linear(deep_input_dims, neuron_num))
            self.layers.append(nn.ReLU())
            deep_input_dims = neuron_num

        self.layers.append(nn.Linear(deep_input_dims, self.reward_dims))

        weight_init([self.layers[-1]])

        self.mlp = nn.Sequential(*self.layers)

    def forward(self, input):
        actions_value = self.mlp(self.bn_input(input))
        return actions_value


class RewardNet:
    def __init__(
            self,
            neuron_dims,
            action_dims,
            reward_dims,
            state_dims,
            lr=0.001,
            memory_size=500,
            batch_size=32,
            device='cuda:0',
    ):
        self.neuron_dims = neuron_dims
        self.action_dims = action_dims
        self.reward_dims = reward_dims
        self.state_dims = state_dims
        self.lr = lr
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.device = device

        # 设置随机数种子
        setup_seed(1)

        if not hasattr(self, 'memory_S_counter'):
            self.memory_S_counter = 0

        if not hasattr(self, 'memory_D_counter'):
            self.memory_D_counter = 0

        # 将经验池<状态-动作-累积奖励>中的转换组初始化为0
        self.memory_S = defaultdict()

        # 将经验池<状态-动作-累积奖励中最大>中的转换组初始化为0
        self.memory_D = torch.zeros((self.memory_size, self.state_dims + 2))

        self.model_reward = Net(self.state_dims, self.action_dims, self.reward_dims, self.neuron_dims).to(
            self.device)
        self.real_reward = Net(self.state_dims, self.action_dims, self.reward_dims, self.neuron_dims).to(self.device)
        self.real_reward.load_state_dict(self.model_reward.state_dict())

        # 优化器
        self.optimizer = torch.optim.Adam(self.model_reward.parameters(), lr=self.lr, weight_decay=1e-5)

    def return_model_reward(self, state_action):
        # 统一 observation 的 shape (1, size_of_observation)
        self.model_reward.eval()
        with torch.no_grad():
            model_reward = self.model_reward.forward(state_action).item()

        return model_reward

    def store_S_pair(self, state_action_pair, reward):
        self.memory_S[state_action_pair] = reward

    def get_reward_from_S(self, state_action_pair):
        return self.memory_S.get(state_action_pair, 0)

    def store_D_pair(self, state_action_reward_pair):
        index = self.memory_D_counter % self.memory_size
        self.memory_D[index, :] = state_action_reward_pair
        self.memory_D_counter += 1

    def learn(self):
        if self.memory_D_counter > self.memory_size:
            sample_index = random.sample(range(self.memory_size), self.batch_size)
        else:
            sample_index = random.sample(range(self.memory_D_counter), self.batch_size)

        batch_memory = self.memory_D[sample_index, :]

        state_actions = batch_memory[:, :self.state_dims + 1].to(self.device)
        real_reward = batch_memory[:, self.state_dims + 1].unsqueeze(1).to(self.device)

        model_reward = self.model_reward.forward(state_actions)

        loss = F.mse_loss(model_reward, real_reward)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model_reward.parameters(), max_norm=40, norm_type=2)
        self.optimizer.step()