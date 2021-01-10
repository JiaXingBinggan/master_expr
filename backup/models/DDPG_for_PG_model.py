import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy

from src.models.Feature_embedding import Feature_Embedding

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(1)


class Actor(nn.Module):
    def __init__(self, input_dims, action_nums, feature_nums, field_nums, latent_dims):
        super(Actor, self).__init__()
        self.input_dims = input_dims

        self.bn_input = nn.BatchNorm1d(1)
        # nn.init.xavier_uniform_(self.bn_input.weight)

        deep_input_dims = self.input_dims + 1
        layers = list()
        neuron_nums = [300, 300, 300]
        for neuron_num in neuron_nums:
            layers.append(nn.Linear(deep_input_dims, neuron_num))
            layers.append(nn.BatchNorm1d(neuron_num))
            layers.append(nn.ReLU())
            deep_input_dims = neuron_num

        # for i, layer in enumerate(layers):
        #     if i % 3 == 0:
        #         nn.init.xavier_uniform_(layer.weight)
        layers.append(nn.Linear(deep_input_dims, action_nums))

        self.mlp = nn.Sequential(*layers)

    def forward(self, input, ddqn_a):
        obs = torch.cat([input, self.bn_input(ddqn_a)], dim=1)

        out = torch.softmax(self.mlp(obs), dim=1)

        return out


class Critic(nn.Module):
    def __init__(self, input_dims, action_nums, feature_nums, field_nums, latent_dims):
        super(Critic, self).__init__()

        self.bn_input = nn.BatchNorm1d(1)
        # nn.init.xavier_uniform_(self.bn_input.weight)
        deep_input_dims = input_dims + action_nums + 1
        layers = list()

        neuron_nums = [300, 300, 300]
        for neuron_num in neuron_nums:
            layers.append(nn.Linear(deep_input_dims, neuron_num))
            layers.append(nn.BatchNorm1d(neuron_num))
            layers.append(nn.ReLU())
            deep_input_dims = neuron_num

        # for i, layer in enumerate(layers):
        #     if i % 3 == 0:
        #         nn.init.xavier_uniform_(layer.weight)
        layers.append(nn.Linear(deep_input_dims, action_nums))

        self.mlp = nn.Sequential(*layers)

    def forward(self, input, action, ddqn_a):
        obs = torch.cat([input, self.bn_input(ddqn_a)], dim=1)
        cat = torch.cat([obs, action], dim=1)

        q_out = self.mlp(cat)

        return q_out

class DDPG():
    def __init__(
            self,
            feature_nums,
            field_nums=15,
            latent_dims=5,
            action_nums=2,
            campaign_id='1458',
            lr_A=1e-4,
            lr_C=1e-3,
            reward_decay=1,
            memory_size=4096000,
            batch_size=256,
            tau=0.005, # for target network soft update
            device='cuda:0',
    ):
        self.feature_nums = feature_nums
        self.field_nums = field_nums
        self.action_nums = action_nums
        self.campaign_id = campaign_id
        self.lr_A = lr_A
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
        self.memory_action_reward = torch.zeros(size=[self.memory_size, self.action_nums + 1]).to(self.device)
        self.memory_ddqn_action = torch.zeros(size=[self.memory_size, 1]).to(self.device)

        self.Actor = Actor(self.input_dims, self.action_nums, self.feature_nums, self.field_nums, self.latent_dims).to(self.device)
        self.Critic = Critic(self.input_dims, self.action_nums, self.feature_nums, self.field_nums, self.latent_dims).to(self.device)

        self.Actor_ = Actor(self.input_dims, self.action_nums, self.feature_nums, self.field_nums, self.latent_dims).to(self.device)
        self.Critic_ = Critic(self.input_dims, self.action_nums, self.feature_nums, self.field_nums, self.latent_dims).to(self.device)

        # 优化器
        self.optimizer_a = torch.optim.Adam(self.Actor.parameters(), lr=self.lr_A, weight_decay=1e-5)
        self.optimizer_c = torch.optim.Adam(self.Critic.parameters(), lr=self.lr_C, weight_decay=1e-5)

        self.loss_func = nn.MSELoss(reduction='mean')

    def store_transition(self, features, action_rewards, ddqn_actions):
        transition_lens = len(features)

        # 由于已经定义了经验池的memory_size，如果超过此大小，旧的memory则被新的memory替换
        index_start = self.memory_counter % self.memory_size
        index_end = (self.memory_counter + transition_lens) % self.memory_size

        if index_end > index_start:
            self.memory_state[index_start: index_end, :] = features  # 替换
            self.memory_action_reward[index_start: index_end, :] = action_rewards
            self.memory_ddqn_action[index_start: index_end, :] = ddqn_actions
        else:
            replace_len_1 = self.memory_size - index_start
            self.memory_state[index_start: self.memory_size, :] = features[0: replace_len_1]
            self.memory_action_reward[index_start: self.memory_size, :] = action_rewards[0: replace_len_1]
            self.memory_ddqn_action[index_start: self.memory_size, :] = ddqn_actions[0: replace_len_1]

            replace_len_2 = transition_lens - replace_len_1
            self.memory_state[0: replace_len_2, :] = features[replace_len_1: transition_lens]
            self.memory_action_reward[0: replace_len_2, :] = action_rewards[replace_len_1: transition_lens]
            self.memory_ddqn_action[0: replace_len_2, :] = ddqn_actions[replace_len_1: transition_lens]

        self.memory_counter += transition_lens

    # def paramter_noise(self, new_actor, exploration_rate):
    #     new_actor.bn_input.weight.data += torch.normal(0, exploration_rate, size=new_actor.bn_input.weight.data.size()).to(self.device)
    #     for i, layer in enumerate(new_actor.mlp):
    #         if i % 3 == 0 or (i - 1) % 3 == 0:
    #             layer.weight.data += torch.normal(0, exploration_rate, size=layer.weight.data.size()).to(self.device)
    #
    #     return new_actor
    #
    # def choose_action(self, state, ddqn_a, exploration_rate):
    #
    #     # state = self.embedding_layer.forward(state)
    #     new_actor = self.paramter_noise(copy.deepcopy(self.Actor), exploration_rate)
    #
    #     new_actor.eval()
    #     with torch.no_grad():
    #         action = new_actor.forward(state, ddqn_a)
    #     # print(action)
    #     return action

    def choose_action(self, state, ddqn_a, exploration_rate):

        # state = self.embedding_layer.forward(state)

        self.Actor.eval()
        with torch.no_grad():
            action = self.Actor.forward(state, ddqn_a)
        self.Actor.train()

        random_seeds = torch.rand(len(state), 1).to(self.device)

        random_action = torch.softmax(torch.normal(action, exploration_rate), dim=1)

        # exploration_rate = max(exploration_rate, 0.1)
        actions = torch.where(random_seeds >= exploration_rate, action,
                              random_action)

        return actions

    def choose_best_action(self, state, ddqn_a):
        # state = self.embedding_layer.forward(state)

        self.Actor.eval()
        with torch.no_grad():
            action = self.Actor.forward(state, ddqn_a)

        return action, torch.softmax(action, dim=1)

    def soft_update(self, net, net_target):
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def sample_batch(self):
        if self.memory_counter > self.memory_size:
            # replacement 代表的意思是抽样之后还放不放回去，如果是False的话，那么出来的三个数都不一样，如果是True的话， 有可能会出现重复的，因为前面的抽的放回去了
            sample_index = torch.LongTensor(random.sample(range(self.memory_size), self.batch_size)).to(self.device)
        else:
            sample_index = torch.LongTensor(random.sample(range(self.memory_counter), self.batch_size)).to(self.device)

        batch_memory_states = self.memory_state[sample_index, :].long()
        batch_memory_action_rewards = self.memory_action_reward[sample_index, :]
        batch_memory_ddqn_actions = self.memory_ddqn_action[sample_index, :]

        # b_s = self.embedding_layer.forward(batch_memory_states)
        b_s = batch_memory_states
        b_a = batch_memory_action_rewards[:, 0: self.action_nums]
        b_r = torch.unsqueeze(batch_memory_action_rewards[:, self.action_nums], 1)
        # b_s_ = self.embedding_layer.forward(batch_memory_states)
        b_s_ = batch_memory_states

        return b_s, b_a, b_r, b_s_, batch_memory_ddqn_actions

    def learn_c(self, b_s, b_a, b_r, b_s_, b_ddqn_a):
        q_target = b_r + self.gamma * self.Critic_.forward(b_s_, self.Actor_.forward(b_s_, b_ddqn_a), b_ddqn_a).detach()
        q = self.Critic.forward(b_s, b_a, b_ddqn_a)

        td_error = self.loss_func(q, q_target)

        self.optimizer_c.zero_grad()
        td_error.backward()
        self.optimizer_c.step()

        td_error_r = td_error.item()

        return td_error_r

    def learn_a(self, b_s, b_ddqn_a):

        a_loss = -self.Critic.forward(b_s, self.Actor.forward(b_s, b_ddqn_a), b_ddqn_a).mean()

        self.optimizer_a.zero_grad()
        a_loss.backward()
        self.optimizer_a.step()

        a_loss_r = a_loss.item()

        return a_loss_r

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
