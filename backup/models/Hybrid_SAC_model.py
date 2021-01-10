import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import datetime
import copy
from torch.distributions import Normal, Categorical

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
        self.epsilon = 1e-3 # 防止出现zero priority
        self.alpha = 0.6 # 取值范围(0,1)，表示td error对priority的影响
        self.beta = 0.4 # important sample， 从初始值到1
        self.beta_increment_per_sampling = 1e-5
        self.abs_err_upper = 1 # abs_err_upper和epsilon ，表明p优先级值的范围在[epsilon,abs_err_upper]之间，可以控制也可以不控制

        self.memory_size = memory_size
        self.memory_counter = 0

        self.priorities_ = torch.zeros(size=[memory_size, 1]).to(self.device)
        # indexs = torch.range(0, self.memory_size)
        # self.prioritys_[:, 1] = indexs

        self.memory = torch.zeros(size=[memory_size, transition_lens]).to(self.device)

    def get_priority(self, td_error):
        return torch.pow(torch.abs(td_error) + self.epsilon, self.alpha)

    def add(self, transitions): # td_error是tensor矩阵
        transition_lens = len(transitions)
        temp_priorities = torch.ones(size=[transition_lens, 1]).to(self.device)

        memory_start = self.memory_counter % self.memory_size
        memory_end = (self.memory_counter + len(transitions)) % self.memory_size

        if memory_end > memory_start:
            self.memory[memory_start: memory_end, :] = transitions
            self.priorities_[memory_start: memory_end, :] = torch.max(self.priorities_[memory_start: memory_end, :], temp_priorities)
        else:
            replace_len_1 = self.memory_size - memory_start
            self.memory[memory_start: self.memory_size, :] = transitions[0: replace_len_1]
            self.priorities_[memory_start: self.memory_size, :] = torch.max(self.priorities_[memory_start: self.memory_size, :], temp_priorities[0: replace_len_1, :])

            replace_len_2 = transition_lens - replace_len_1
            self.memory[:replace_len_2, :] = transitions[replace_len_1: transition_lens]
            self.priorities_[:replace_len_2, :] = torch.max(self.priorities_[:replace_len_2, :], temp_priorities[replace_len_1: transition_lens, :])

        self.memory_counter += len(transitions)

    def stochastic_sample(self, batch_size):
        total_p = torch.sum(self.priorities_, dim=0)

        if self.memory_counter >= self.memory_size:
            min_prob = torch.min(self.priorities_)
            # 采样概率分布
            P = torch.div(self.priorities_, total_p).squeeze(1).cpu().numpy()
            sample_indexs = torch.Tensor(np.random.choice(self.memory_size, batch_size, p=P, replace=False)).long().to(self.device)
        else:
            min_prob = torch.min(self.priorities_[:self.memory_counter, :])
            P = torch.div(self.priorities_[:self.memory_counter, :], total_p).squeeze(1).cpu().numpy()
            sample_indexs = torch.Tensor(np.random.choice(self.memory_counter, batch_size, p=P, replace=False)).long().to(self.device)

        self.beta = torch.min(torch.FloatTensor([1., self.beta + self.beta_increment_per_sampling])).item()

        batch = self.memory[sample_indexs]
        choose_priorities = self.priorities_[sample_indexs]
        ISweights = torch.pow(torch.div(choose_priorities, min_prob), -self.beta)

        return sample_indexs, batch, ISweights

    def greedy_sample(self, batch_size):
        # total_p = torch.sum(self.prioritys_, dim=0)

        if self.memory_counter >= self.memory_size:
            min_prob = torch.min(self.priorities_)
        else:
            min_prob = torch.min(self.priorities_[:self.memory_counter, :])
        self.beta = torch.min(torch.FloatTensor([1., self.beta + self.beta_increment_per_sampling])).item()

        sorted_priorities, sorted_indexs = torch.sort(-self.priorities_, dim=0)

        choose_idxs = sorted_indexs[:batch_size, :].squeeze(1)

        batch = self.memory[choose_idxs]

        choose_priorities = -sorted_priorities[:batch_size, :]

        ISweights = torch.pow(torch.div(choose_priorities, min_prob), -self.beta).detach()

        return choose_idxs, batch, ISweights

    def batch_update(self, choose_idx, td_errors):
        p = self.get_priority(td_errors)
        self.priorities_[choose_idx, :] = p


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


class C_Actor(nn.Module): # Gaussion Policy
    def __init__(self, input_dims, action_nums):
        super(C_Actor, self).__init__()
        self.input_dims = input_dims
        self.action_nums = action_nums

        self.bn_input = nn.BatchNorm1d(self.input_dims)

        hidden_dims = [256, 256]
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dims, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU()
        )
        # self.mlp_l1 = nn.Linear(self.input_dims, hidden_dims[0])
        # self.mlp_l2 = nn.Linear(hidden_dims[0], hidden_dims[1])

        self.mean_linear = nn.Linear(hidden_dims[1], self.action_nums)
        self.log_std_linear = nn.Linear(hidden_dims[1], self.action_nums)

        # self.apply(weights_init_)

    def forward(self, state):
        # x = F.relu(self.mlp_l1(state))
        # x = F.relu(self.mlp_l2(x))

        x = self.mlp(self.bn_input(state))

        action_mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=-20, max=2) # interval

        return action_mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        x_t = normal.rsample() # rsample()不是在定义的正太分布上采样，而是先对标准正太分布N(0,1)N(0,1)进行采样，然后输出：mean+std×采样值
        y_t = torch.tanh(x_t)
        actions = y_t # 返回的连续型动作

        # Enforcing Action Bound
        log_prob = (normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)).sum(-1, keepdim=True)

        return actions, log_prob

    def evaluate(self, state):
        mean, log_std = self.forward(state)
        actions = torch.tanh(mean)

        return actions


class D_Actor(nn.Module):
    def __init__(self, input_dims, action_dims):
        super(D_Actor, self).__init__()
        self.input_dims = input_dims
        self.action_dims = action_dims

        self.bn_input = nn.BatchNorm1d(self.input_dims)
        hidden_dims = [256, 256]
        self.mlp_l1 = nn.Linear(self.input_dims, hidden_dims[0])
        self.mlp_l2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        # self.mlp = nn.Sequential(
        #     nn.Linear(self.input_dims, hidden_dims[0]),
        #     nn.BatchNorm1d(hidden_dims[0]),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dims[0], hidden_dims[1]),
        #     nn.BatchNorm1d(hidden_dims[1]),
        #     nn.ReLU()
        # )

        self.policy_layer = nn.Linear(hidden_dims[1], self.action_dims)

        # self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.mlp_l1(state))
        x = F.relu(self.mlp_l2(x))
        # x = self.mlp(self.bn_input(state))

        action_logits = F.softmax(self.policy_layer(x), dim=-1)

        return action_logits

    def sample(self, state):
        action_probs = self.forward(state)

        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1) + 1

        # avoid numerical instability
        mirror = (action_probs == 0.0).float() * 1e-6

        log_action_probs = torch.log(action_probs + mirror)

        return actions, action_probs, log_action_probs

    def evaluate(self, state):
        action_probs = self.forward(state)

        actions = torch.argmax(action_probs, dim=-1, keepdim=True) + 1

        return actions


class Hybrid_Q_network(nn.Module):
    def __init__(self, input_dims, action_dims):
        super(Hybrid_Q_network, self).__init__()
        self.input_dims = input_dims
        self.action_dims = action_dims

        # Q1
        # self.bn_input = nn.BatchNorm1d(self.input_dims)
        hidden_dims = [256, 256]
        self.mlp_q1_l1 = nn.Linear(self.input_dims, hidden_dims[0])
        self.mlp_q1_l2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        # self.mlp_1 = nn.Sequential(
        #     nn.Linear(self.input_dims, hidden_dims[0]),
        #     nn.BatchNorm1d(hidden_dims[0]),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dims[0], hidden_dims[1]),
        #     nn.BatchNorm1d(hidden_dims[1])
        # )

        self.c_q1 = nn.Linear(hidden_dims[1] + self.action_dims, 1)
        self.d_q1 = nn.Linear(hidden_dims[1], self.action_dims)

        # Q2
        self.mlp_q2_l1 = nn.Linear(self.input_dims, hidden_dims[0])
        self.mlp_q2_l2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        # self.mlp_2 = nn.Sequential(
        #     nn.Linear(self.input_dims, hidden_dims[0]),
        #     nn.BatchNorm1d(hidden_dims[0]),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dims[0], hidden_dims[1]),
        #     nn.BatchNorm1d(hidden_dims[1])
        # )

        self.c_q2 = nn.Linear(hidden_dims[1] + self.action_dims, 1)
        self.d_q2 = nn.Linear(hidden_dims[1], self.action_dims)

        # self.apply(weights_init_)

    def forward(self, state, action):
        # bn_state = self.bn_input(state)
        x1 = F.relu(self.mlp_q1_l1(state))
        x1 = F.relu(self.mlp_q1_l2(x1))
        # x1 = self.mlp_1(bn_state)
        c_q1 = self.c_q1(torch.cat([x1, action], dim=-1))
        d_q1 = self.d_q1(x1)

        x2 = F.relu(self.mlp_q2_l1(state))
        x2 = F.relu(self.mlp_q2_l2(x2))
        #
        # x2 = self.mlp_2(bn_state)
        c_q2 = self.c_q2(torch.cat([x2, action], dim=-1))
        d_q2 = self.d_q2(x2)

        return c_q1, d_q1, c_q2, d_q2


class Hybrid_RL_Model():
    def __init__(
            self,
            feature_nums,
            field_nums=15,
            latent_dims=5,
            action_nums=2,
            campaign_id='1458',
            lr_C_A=3e-4,
            lr_D_A=3e-4,
            lr_C=3e-4,
            reward_decay=1,
            memory_size=4096000,
            batch_size=256,
            tau=0.005,  # for target network soft update
            device='cuda:0',
    ):
        self.feature_nums = feature_nums
        self.field_nums = field_nums
        self.action_nums = action_nums
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

        self.memory = Memory(self.memory_size, self.field_nums + self.action_nums + 2, self.device)

        self.Critic = Hybrid_Q_network(self.input_dims, self.action_nums).to(self.device)
        self.Critic_ = copy.deepcopy(self.Critic)

        self.D_Actor = D_Actor(self.input_dims, self.action_nums).to(self.device)

        self.C_Actor = C_Actor(self.input_dims, self.action_nums).to(self.device)

        # 优化器
        self.optimizer_c_a = torch.optim.Adam(self.C_Actor.parameters(), lr=self.lr_C_A, eps=1e-8, weight_decay=1e-2)
        self.optimizer_d_a = torch.optim.Adam(self.D_Actor.parameters(), lr=self.lr_D_A, eps=1e-8, weight_decay=1e-2)
        self.optimizer_c = torch.optim.Adam(self.Critic.parameters(), lr=self.lr_C, eps=1e-8, weight_decay=1e-2)

        # automatic entropy tuning
        self.c_target_entropy = -torch.prod(torch.Tensor([self.action_nums, 1]).to(self.device)).item()
        self.c_log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.c_alpha = self.c_log_alpha.exp()
        self.optimizer_c_alpha = torch.optim.Adam([self.c_log_alpha], lr=lr_C, eps=1e-8, weight_decay=1e-2)

        # Target entropy is -log(1/|A|) * ratio (= maximum entropy * ratio)
        target_entropy_ratio = 0.98
        self.d_target_entropy = -np.log(1.0 / self.action_nums) * target_entropy_ratio
        self.d_log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.d_alpha = self.d_log_alpha.exp()
        self.optimizer_d_alpha = torch.optim.Adam([self.d_log_alpha], lr=lr_C, eps=1e-8, weight_decay=1e-2)

        self.learn_iter = 0

    def store_transition(self, transitions): # 所有的值都应该弄成float
        self.memory.add(transitions)

    def choose_action(self, state):
        with torch.no_grad():
            c_actions, _ = self.C_Actor.sample(state)
            d_actions, _, _ = self.D_Actor.sample(state)

        ensemble_c_actions = torch.softmax(c_actions, dim=-1)

        return c_actions, ensemble_c_actions, d_actions

    def choose_best_action(self, state):
        with torch.no_grad():
            c_actions = self.C_Actor.evaluate(state)
            d_actions = self.D_Actor.evaluate(state)

        ensemble_c_actions = torch.softmax(c_actions, dim=-1)

        return ensemble_c_actions, d_actions

    def soft_update(self, net, net_target):
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def hard_update(self, net, net_target):
        net_target.load_state_dict(net.state_dict())

    def learn(self, embedding_layer):
        self.learn_iter += 1

        # sample
        choose_idx, batch_memory, ISweights = self.memory.stochastic_sample(self.batch_size)

        b_s = embedding_layer.forward(batch_memory[:, :self.field_nums].long())
        b_c_a = batch_memory[:, self.field_nums: self.field_nums + self.action_nums]
        b_discrete_a = torch.unsqueeze(batch_memory[:, self.field_nums + self.action_nums] - 1, 1).long()
        b_r = torch.unsqueeze(batch_memory[:, -1], 1)
        b_s_ = b_s  # embedding_layer.forward(batch_memory_states)

        with torch.no_grad():
            c_action_next, c_log_probs_next = self.C_Actor.sample(b_s_)
            d_action_next, d_action_probs_next, d_log_probs_next = self.D_Actor.sample(b_s_)

            c_q1_next, d_q1_next, c_q2_next, d_q2_next = self.Critic_.forward(b_s_, c_action_next)

            q_c_next_target = b_r + self.gamma * (torch.min(c_q1_next, c_q2_next) - self.c_alpha * c_log_probs_next)

            q_d_next_target = (b_r + self.gamma * (torch.min(d_q1_next, d_q2_next) - self.d_alpha * d_log_probs_next) * d_action_probs_next).mean(dim=-1).unsqueeze(-1)

        c_q1, d_q1, c_q2, d_q2 = self.Critic.forward(b_s, b_c_a)

        # c_actor's criric loss
        c_critic_loss = ISweights * ((c_q1 - q_c_next_target).pow(2) + (c_q2 - q_c_next_target).pow(2))
        # c_critic_loss = F.mse_loss(c_q1, q_c_next_target) + F.mse_loss(c_q2, q_c_next_target)

        # d_actor's critic loss
        d_critic_loss = ISweights * ((d_q1.gather(1, b_discrete_a) - q_d_next_target).pow(2) + (d_q2.gather(1, b_discrete_a) - q_d_next_target).pow(2))
        # d_critic_loss = F.mse_loss(d_q1.gather(1, b_discrete_a), q_d_next_target) + F.mse_loss(d_q2.gather(1, b_discrete_a), q_d_next_target)

        critic_loss = (c_critic_loss + d_critic_loss).mean()

        self.optimizer_c.zero_grad()
        critic_loss.backward()
        self.optimizer_c.step()

        td_errors = ((2 * q_c_next_target - c_q1 - c_q2) / 2 + 1e-6) + ((2 * q_d_next_target - d_q1.gather(1, b_discrete_a) - d_q2.gather(1, b_discrete_a)) / 2 + 1e-6)
        self.memory.batch_update(choose_idx, td_errors.detach())

        c_action, c_log_probs = self.C_Actor.sample(b_s)
        d_action, d_action_probs, d_log_probs = self.D_Actor.sample(b_s)

        c_q1, d_q1, c_q2, d_q2 = self.Critic.forward(b_s, b_c_a)
        # c_actor's loss

        c_actor_loss = (ISweights * (self.c_alpha * c_log_probs - torch.min(c_q1, c_q2))).mean()
        # c_actor_loss = (self.c_alpha * c_log_probs - torch.min(c_q1, c_q2)).mean()
        self.optimizer_c_a.zero_grad()
        c_actor_loss.backward(retain_graph=True)
        self.optimizer_c_a.step()

        c_entropies = c_log_probs

        # d_actor's loss
        d_actor_loss = (ISweights * (d_action_probs * (self.d_alpha * d_log_probs - torch.min(d_q1, d_q2)))).mean()
        # d_actor_loss = (d_action_probs * (self.d_alpha * d_log_probs - torch.min(d_q1, d_q2))).mean()
        self.optimizer_d_a.zero_grad()
        d_actor_loss.backward(retain_graph=True)
        self.optimizer_d_a.step()

        d_entropies = torch.sum(d_action_probs * d_log_probs, dim=-1)

        # c_actor's entropy tuning loss
        c_aplha_loss = -(self.c_log_alpha * (c_entropies + self.c_target_entropy)).mean()
        self.optimizer_c_alpha.zero_grad()
        c_aplha_loss.backward(retain_graph=True)
        self.optimizer_c_alpha.step()

        self.c_alpha = self.c_log_alpha.exp()

        # d_actor's entropy tuning loss
        d_alpha_loss = -(self.d_log_alpha * (d_entropies + self.d_target_entropy)).mean()
        self.optimizer_d_alpha.zero_grad()
        d_alpha_loss.backward(retain_graph=True)
        self.optimizer_d_alpha.step()

        self.d_alpha = self.d_log_alpha.exp()

        if self.learn_iter % 100 == 0:
            self.hard_update(self.Critic, self.Critic_)
        # self.soft_update(self.Critic, self.Critic_)

        return critic_loss.item()

