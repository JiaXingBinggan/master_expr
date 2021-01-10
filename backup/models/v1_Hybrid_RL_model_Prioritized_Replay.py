import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.distributions import MultivariateNormal, Categorical
import datetime
from torch.distributions import Normal, Categorical

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(1)

class SumTree(object):
    def __init__(self, nodes, transition_lens, device): # transition_lens = feature_nums + c_a_nums + d_a_nums + reward_nums
        self.nodes = nodes # leaf nodes
        self.device = device

        self.sum_tree = torch.zeros(size=[2 * self.nodes - 1, 1]).to(self.device) # parents nodes = nodes - 1

        self.data = torch.zeros(size=[self.nodes, transition_lens]).to(self.device)

        self.data_pointer = 0

    def add_leaf(self, p, transitions): # p-priority
        tree_idx_start = self.data_pointer + self.nodes - 1 #python从0开始索引，初始时tree_idx表示第一个叶节点的索引值，样本按叶子结点依次向后排
        tree_idx_end = tree_idx_start + len(transitions)
        self.data[self.data_pointer: self.data_pointer + len(transitions), :] = transitions
        self.update(tree_idx_start, tree_idx_end, p)

        self.data_pointer += len(transitions)

        if self.data_pointer >= self.nodes:
            self.data_pointer = 0

    def update(self, tree_idx_start, tree_idx_end, p):
        changes = p - self.sum_tree[tree_idx_start: tree_idx_end]
        self.sum_tree[tree_idx_start: tree_idx_end] = p

        temp_tree_idxs = np.arange(tree_idx_start, tree_idx_end)
        for i, tree_idx in enumerate(temp_tree_idxs):
            while tree_idx != 0:
                tree_idx = (tree_idx - 1) // 2
                self.sum_tree[tree_idx] += changes[i] # 这里要改，因为parents_tree_idx会有相同的部分

    def get_leaf(self, v): # v-随机选择的p值，用于抽取data， 只有一条
        parent_idx = 0

        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1

            if cl_idx >= len(self.sum_tree): # 没有子节点了
                leaf_idx = parent_idx
                break
            else:
                if v <= self.sum_tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.sum_tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.nodes + 1 # 减去非叶子节点索引数

        return leaf_idx, self.sum_tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.sum_tree[0] # root's total priority

class Memory(object):
    def __init__(self, memory_size, transition_lens, device):
        self.device = device
        self.transition_lens = transition_lens # 存储的数据长度
        self.epsilon = 1e-3 # 防止出现zero priority
        self.alpha = 0.6 # 取值范围(0,1)，表示td error对priority的影响
        self.beta = 0.4 # important sample， 从初始值到1
        self.beta_increment_per_sampling = 0.0001
        self.abs_err_upper = 1 # abs_err_upper和epsilon ，表明p优先级值的范围在[epsilon,abs_err_upper]之间，可以控制也可以不控制

        self.memory_size = memory_size
        self.memory_counter = 0

        self.prioritys_ = torch.zeros(size=[memory_size, 1]).to(self.device)
        # indexs = torch.range(0, self.memory_size)
        # self.prioritys_[:, 1] = indexs

        self.memory = torch.zeros(size=[memory_size, transition_lens]).to(self.device)

    def get_priority(self, td_error):
        return torch.pow(torch.abs(td_error) + self.epsilon, self.alpha)

    def add(self, td_error, transitions): # td_error是tensor矩阵
        transition_lens = len(transitions)
        p = self.get_priority(td_error)

        memory_start = self.memory_counter % self.memory_size
        memory_end = (self.memory_counter + len(transitions)) % self.memory_size

        if memory_end > memory_start:
            self.memory[memory_start: memory_end, :] = transitions
            self.prioritys_[memory_start: memory_end, :] = p
        else:
            replace_len_1 = self.memory_size - memory_start
            self.memory[memory_start: self.memory_size, :] = transitions[0: replace_len_1]
            self.prioritys_[memory_start: self.memory_size, :] = p[0: replace_len_1]

            replace_len_2 = transition_lens - replace_len_1
            self.memory[:replace_len_2, :] = transitions[replace_len_1: transition_lens]
            self.prioritys_[:replace_len_2, :] = p[replace_len_1: transition_lens]

        self.memory_counter += len(transitions)

    def stochastic_sample(self, batch_size):
        total_p = torch.sum(self.prioritys_, dim=0)

        if self.memory_counter >= self.memory_size:
            min_prob = torch.min(self.prioritys_)
            # 采样概率分布
            P = torch.div(self.prioritys_, total_p).squeeze(1).cpu().numpy()
            sample_indexs = torch.Tensor(np.random.choice(self.memory_size, batch_size, p=P)).long().to(self.device)
        else:
            min_prob = torch.min(self.prioritys_[:self.memory_counter, :])
            P = torch.div(self.prioritys_[:self.memory_counter, :], total_p).squeeze(1).cpu().numpy()
            sample_indexs = torch.Tensor(np.random.choice(self.memory_counter, batch_size, p=P)).long().to(self.device)

        self.beta = torch.min(torch.FloatTensor([1., self.beta + self.beta_increment_per_sampling])).item()

        batch = self.memory[sample_indexs]
        choose_priorities = self.prioritys_[sample_indexs]
        ISweights = torch.pow(torch.div(choose_priorities, min_prob), -self.beta)

        return sample_indexs, batch, ISweights

    def greedy_sample(self, batch_size):
        # total_p = torch.sum(self.prioritys_, dim=0)

        if self.memory_counter >= self.memory_size:
            min_prob = torch.min(self.prioritys_)
        else:
            min_prob = torch.min(self.prioritys_[:self.memory_counter, :])
        self.beta = torch.min(torch.FloatTensor([1., self.beta + self.beta_increment_per_sampling])).item()

        sorted_priorities, sorted_indexs = torch.sort(-self.prioritys_, dim=0)

        choose_idxs = sorted_indexs[:batch_size, :].squeeze(1)

        batch = self.memory[choose_idxs]

        choose_priorities = -sorted_priorities[:batch_size, :]

        ISweights = torch.pow(torch.div(choose_priorities, min_prob), -self.beta).detach()

        # if self.memory_counter >= self.memory_size:
        #     sample_index = torch.LongTensor(random.sample(range(self.memory_size), self.batch_size)).to(self.device)
        # else:
        #     sample_index = torch.LongTensor(random.sample(range(self.memory_counter), self.batch_size)).to(self.device)
        return choose_idxs, batch, ISweights

    def batch_update(self, choose_idx, td_errors):
        p = self.get_priority(td_errors)
        self.prioritys_[choose_idx, :] = p

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
            nn.Linear(neuron_nums[2], action_nums),
            nn.Softmax(dim=-1)
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

        q_out = self.mlp(cat)

        return q_out

class multi_models(nn.Module):
    def __init__(self, input_dims, action_nums):
        super(multi_models, self).__init__()
        self.input_dims = input_dims
        self.c_action_dims = action_nums
        self.d_action_dims = action_nums - 1

        self.bn_input = nn.BatchNorm1d(self.input_dims)

        neuron_nums = [300, 300, 300]
        self.feature_exact_layers = nn.Sequential(
            nn.Linear(self.input_dims, neuron_nums[0]),
            nn.BatchNorm1d(neuron_nums[0]),
            nn.ReLU(),
            nn.Linear(neuron_nums[0], neuron_nums[1]),
            nn.BatchNorm1d(neuron_nums[1]),
            nn.ReLU(),
            nn.Linear(neuron_nums[1], neuron_nums[2]),
            nn.BatchNorm1d(neuron_nums[2])
        )# 特征提取层

        self.critic_layer = nn.Linear(neuron_nums[2] + self.c_action_dims, 1)
        self.c_action_layer = nn.Sequential(
            nn.Linear(neuron_nums[2], self.c_action_dims),
            nn.Tanh()
        )
        self.d_action_layer = nn.Sequential(
            nn.Linear(neuron_nums[2], self.d_action_dims),
            nn.Softmax(dim=-1)
        )

        self.c_action_std = torch.ones(size=[1, self.c_action_dims])

    def act(self, input):
        obs = self.bn_input(input)

        mlp_out = self.mlp(obs)

        c_actions_means = self.c_action_layer(mlp_out)
        c_action_dist = Normal(c_actions_means, F.softplus(self.action_std))
        c_actions = torch.clamp(c_action_dist.sample(), -1, 1) # 用于返回训练
        ensemble_c_actions = torch.softmax(c_actions, dim=-1)

        d_actions_q_values = self.d_action_layer(mlp_out)
        d_action_dist = Categorical(d_actions_q_values)
        d_actions = d_action_dist.sample()
        ensemble_d_actions = d_actions + 2

        return c_actions_means, d_actions_q_values

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
            tau=0.005,  # for target network soft update
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

        self.memory = Memory(self.memory_size, self.field_nums + self.c_a_action_nums + 2, self.device)

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

    def store_transition(self, transitions, embedding_layer): # 所有的值都应该弄成float
        b_s = embedding_layer.forward(transitions[:, :self.field_nums].long())
        b_s_ = b_s
        b_a = transitions[:, self.field_nums: self.field_nums + self.c_a_action_nums]
        b_discrete_a = torch.unsqueeze(transitions[:, self.field_nums + self.c_a_action_nums], 1)
        b_r = torch.unsqueeze(transitions[:, -1], dim=1)

        # critic
        q_target_critic = b_r + self.gamma * self.Critic_.forward(b_s_, self.Continuous_Actor_.forward(b_s_,
                                                                                                b_discrete_a),
                                                           b_discrete_a)
        q_critic = self.Critic.forward(b_s, b_a, b_discrete_a)

        td_error_critic = q_target_critic - q_critic

        # D_A
        q_eval_d_a = self.Discrete_Actor.forward(b_s).gather(1,
                                                         b_discrete_a.long() - 2)  # shape (batch,1), gather函数将对应action的Q值提取出来做Bellman公式迭代
        q_next_d_a = self.Discrete_Actor_.forward(b_s_)
        # q_eval_next = self.Discrete_Actor.forward(b_s_)
        # max_b_a_next = torch.unsqueeze(torch.max(q_eval_next, 1)[1], 1)  # 选择最大Q的动作
        # select_q_next = q_next_d_a.gather(1, max_b_a_next)

        q_target_d_a = b_r + self.gamma * q_next_d_a.max(1)[0].view(-1, 1)  # shape (batch, 1)

        td_error_d_a = q_target_d_a - q_eval_d_a

        td_errors = td_error_critic + td_error_d_a

        self.memory.add(td_errors.detach(), transitions)


    def choose_continuous_action(self, state, discrete_a, exploration_rate):
        self.Continuous_Actor.eval()
        with torch.no_grad():
            action_mean = self.Continuous_Actor.forward(state, discrete_a)
        random_seeds = torch.rand(size=[len(state), 1]).to(self.device)

        # random_action = torch.normal(action_mean, exploration_rate)
        random_action = torch.clamp(torch.normal(action_mean, exploration_rate), -1, 1)

        c_actions = torch.where(random_seeds >= exploration_rate, action_mean, random_action)

        ensemble_c_actions = torch.softmax(c_actions, dim=-1)  # 模型所需的动作

        self.Continuous_Actor.train()

        return c_actions, ensemble_c_actions

    def choose_discrete_action(self, state, exploration_rate):
        self.Discrete_Actor.eval()
        with torch.no_grad():
            action_values = self.Discrete_Actor.forward(state)

        random_seeds = torch.rand(len(state), 1).to(self.device)

        action_dist = Categorical(action_values)
        actions = action_dist.sample()  # 分布产生的结果
        random_d_actions = actions + 2  # 模型所需的动作

        max_d_actions = torch.argsort(-action_values)[:, 0] + 2

        ensemble_d_actions = torch.where(random_seeds >= exploration_rate, max_d_actions.view(-1, 1),
                                         random_d_actions.view(-1, 1))

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
        choose_idx, batch_memory, ISweights = self.memory.stochastic_sample(self.batch_size)

        b_s = embedding_layer.forward(batch_memory[:, :self.field_nums].long())
        b_a = batch_memory[:, self.field_nums: self.field_nums + self.c_a_action_nums]
        b_discrete_a = torch.unsqueeze(batch_memory[:, self.field_nums + self.c_a_action_nums], 1)
        b_r = torch.unsqueeze(batch_memory[:, -1], 1)
        b_s_ = b_s  # embedding_layer.forward(batch_memory_states)

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
        d_a_td_error = (q_target - q_eval).detach()
        d_a_loss = (ISweights * torch.pow(q_eval - q_target.detach(), 2)).mean()

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

        critic_td_error = (q_target - q).detach()
        critic_loss = (ISweights * torch.pow(q - q_target.detach(), 2)).mean()

        self.optimizer_c.zero_grad()
        critic_loss.backward()
        self.optimizer_c.step()

        critic_loss_r = critic_loss.item()

        # C_A
        c_a_loss = -self.Critic.forward(b_s, self.Continuous_Actor.forward(b_s, b_discrete_a), b_discrete_a).mean()

        self.optimizer_c_a.zero_grad()
        c_a_loss.backward()
        self.optimizer_c_a.step()
        c_a_loss_r = c_a_loss.item()

        new_p = d_a_td_error + critic_td_error

        self.memory.batch_update(choose_idx, new_p)

        return critic_loss_r, c_a_loss_r, d_a_loss_r


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
