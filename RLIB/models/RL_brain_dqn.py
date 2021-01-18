import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
from torch.autograd import Variable
from torch.distributions import MultivariateNormal, Categorical
import datetime
from torch.distributions import Normal, Categorical, MultivariateNormal


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Memory(object):
    def __init__(self, memory_size, transition_lens, device):
        self.device = device
        self.transition_lens = transition_lens  # 存储的数据长度
        self.epsilon = 1e-3  # 防止出现zero priority
        self.alpha = 0.6  # 取值范围(0,1)，表示td error对priority的影响
        self.beta = 0.2  # important sample， 从初始值到1
        self.beta_min = 0.4
        self.beta_max = 1.0
        self.beta_increment_per_sampling = 0.00001
        self.abs_err_upper = 1  # abs_err_upper和epsilon ，表明p优先级值的范围在[epsilon,abs_err_upper]之间，可以控制也可以不控制

        self.memory_size = int(memory_size)
        self.memory_counter = 0

        self.prioritys_ = torch.zeros(size=[self.memory_size, 2]).to(self.device)
        # indexs = torch.range(0, self.memory_size)
        # self.prioritys_[:, 1] = indexs

        self.memory = torch.zeros(size=[self.memory_size, transition_lens]).to(self.device)

    def get_priority(self, td_error):
        return torch.pow(torch.abs(td_error) + self.epsilon, self.alpha)

    def add(self, td_error, transitions):  # td_error是tensor矩阵
        transition_lens = len(transitions)
        p = td_error

        memory_start = self.memory_counter % self.memory_size
        memory_end = (self.memory_counter + len(transitions)) % self.memory_size

        if memory_end > memory_start:
            self.memory[memory_start: memory_end, :] = transitions
            self.prioritys_[memory_start: memory_end, :] = p
        else:
            replace_len_1 = self.memory_size - memory_start
            self.memory[memory_start: self.memory_size, :] = transitions[0: replace_len_1]
            self.prioritys_[memory_start: self.memory_size, :] = p[0: replace_len_1, :]

            replace_len_2 = transition_lens - replace_len_1
            self.memory[:replace_len_2, :] = transitions[replace_len_1: transition_lens]
            self.prioritys_[:replace_len_2, :] = p[replace_len_1: transition_lens, :]

        self.memory_counter += len(transitions)

    def stochastic_sample(self, batch_size):
        if self.memory_counter >= self.memory_size:
            priorities = self.get_priority(self.prioritys_[:, 0:1])

            total_p = torch.sum(priorities, dim=0)
            min_prob = torch.min(priorities)
            # 采样概率分布
            P = torch.div(priorities, total_p).squeeze(1).cpu().detach().numpy()
            sample_indexs = torch.Tensor(np.random.choice(self.memory_size, batch_size, p=P, replace=False)).long().to(
                self.device)
        else:
            priorities = self.get_priority(self.prioritys_[:self.memory_counter, 0:1])
            total_p = torch.sum(priorities, dim=0)
            min_prob = torch.min(priorities)
            P = torch.div(priorities, total_p).squeeze(1).cpu().detach().numpy()
            sample_indexs = torch.Tensor(
                np.random.choice(self.memory_counter, batch_size, p=P, replace=False)).long().to(self.device)

        # self.beta = torch.min(torch.FloatTensor([1., self.beta + self.beta_increment_per_sampling])).item()
        # print(self.beta)
        batch = self.memory[sample_indexs]
        choose_priorities = priorities[sample_indexs]
        ISweights = torch.pow(torch.div(choose_priorities, min_prob), -self.beta).detach()

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

        return choose_idxs, batch, ISweights

    def batch_update(self, choose_idx, td_errors):
        # p = self.get_priority(td_errors)
        self.prioritys_[choose_idx, 0:1] = td_errors


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
            layer.weight.data.uniform_(-lim, lim)
            layer.bias.data.fill_(0)


class PolicyNet(nn.Module):
    def __init__(self, input_dims, action_nums):
        super(PolicyNet, self).__init__()
        self.input_dims = input_dims
        self.action_nums = action_nums

        deep_input_dims = self.input_dims
        # self.bn_input = nn.BatchNorm1d(deep_input_dims)
        # self.bn_input.weight.data.fill_(1)
        # self.bn_input.bias.data.fill_(0)

        neuron_nums = [128]

        self.layers = list()
        for neuron_num in neuron_nums:
            self.layers.append(nn.Linear(deep_input_dims, neuron_num))
            self.layers.append(nn.ReLU())
            deep_input_dims = neuron_num

        self.layers.append(nn.Linear(deep_input_dims, self.action_nums))

        weight_init(self.layers)

        self.mlp = nn.Sequential(*self.layers)

    def evaluate(self, input):
        # obs = self.bn_input(input)
        obs = input
        c_q_out = self.mlp(obs)

        return c_q_out

def boltzmann_softmax(actions, temprature):
    return (actions / temprature).exp() / torch.sum((actions / temprature).exp(), dim=-1).view(-1, 1)

def gumbel_softmax_sample(logits, temprature=1.0, hard=False, eps=1e-20, uniform_seed=1.0):
    U = Variable(torch.FloatTensor(*logits.shape).uniform_().cuda(), requires_grad=False)
    y = logits + -torch.log(-torch.log(U + eps) + eps)
    y = F.softmax(y / temprature, dim=-1)

    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y

    return y


def onehot_from_logits(logits, eps=0.0):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    if eps == 0.0:
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    # chooses between best and random actions using epsilon greedy
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])


class DQN():
    def __init__(
            self,
            action_nums=2,
            lr=1e-3,
            reward_decay=1.0,
            memory_size=100000,
            batch_size=32,
            random_seed=1,
            device='cuda:0',
    ):
        self.action_nums = action_nums
        self.lr = lr
        self.gamma = reward_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.device = device

        setup_seed(random_seed)

        self.replace_iter = 100
        self.learn_iter = 0

        self.memory_counter = 0

        self.input_dims = 4

        self.memory = Memory(self.memory_size, self.input_dims * 2 + 3, self.device)

        self.agent = PolicyNet(self.input_dims, action_nums).to(self.device)
        self.agent_ = PolicyNet(self.input_dims, action_nums).to(self.device)
        self.agent_.load_state_dict(self.agent.state_dict())

        # 优化器
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=self.lr, weight_decay=1e-5)

        self.loss_func = nn.MSELoss(reduction='mean')

        self.epsilon = 0.9

    def store_transition(self, transitions):  # 所有的值都应该弄成float
        if torch.max(self.memory.prioritys_) == 0.:
            td_errors = torch.cat(
                [torch.ones(size=[len(transitions), 1]).to(self.device), transitions[:, -1].view(-1, 1)], dim=-1).detach()
        else:
            td_errors = torch.cat(
                [torch.max(self.memory.prioritys_).expand_as(torch.ones(size=[len(transitions), 1])).to(self.device),
                 transitions[:, -1].view(-1, 1)], dim=-1).detach()

        self.memory.add(td_errors, transitions.detach())

    def choose_action(self, state, epsilon):
        self.agent.train()
        with torch.no_grad():
            if random.uniform(0, 1) > epsilon:
                action = random.sample(range(self.action_nums), 1)[0]
            else:
                action = torch.argmax(self.agent.evaluate(state), dim=-1).item()

        return action

    def choose_best_action(self, state):
        self.agent.eval()
        with torch.no_grad():
            action_values = self.agent.evaluate(state)

        return torch.argmax(action_values, dim=-1).item()

    def learn(self):
        self.learn_iter += 1

        if self.learn_iter % self.replace_iter:
            self.agent_.load_state_dict(self.agent.state_dict())

        self.agent.train()
        self.agent_.eval()

        # sample
        choose_idx, batch_memory, ISweights = self.memory.stochastic_sample(self.batch_size)
        # if self.memory.memory_counter > self.memory_size:
        #     # replacement 代表的意思是抽样之后还放不放回去，如果是False的话，那么出来的三个数都不一样，如果是True的话， 有可能会出现重复的，因为前面的抽的放回去了
        #     sample_index = random.sample(range(self.memory_size), self.batch_size)
        # else:
        #     sample_index = random.sample(range(self.memory.memory_counter), self.batch_size)
        #
        # batch_memory = self.memory.memory[sample_index, :]

        b_s = batch_memory[:, :self.input_dims]
        b_a = batch_memory[:, self.input_dims: self.input_dims + 1].long()
        b_s_ = batch_memory[:,
                self.input_dims + 1: self.input_dims * 2 + 1]
        b_done = batch_memory[:, -2].unsqueeze(1)
        b_r = batch_memory[:, -1].unsqueeze(1)

        q_eval = self.agent.evaluate(b_s).gather(1, b_a)  # shape (batch,1), gather函数将对应action的Q值提取出来做Bellman公式迭代
        q_next = self.agent_.evaluate(b_s_).detach()  # detach from graph, don't backpropagate，因为target网络不需要训练

        q_target = b_r + self.gamma * torch.mul(q_next.max(1)[0].view(self.batch_size, 1), b_done)

        td_errors = q_target - q_eval

        self.memory.batch_update(choose_idx, td_errors)

        # 训练eval_net
        loss = (ISweights * F.mse_loss(q_eval, q_target, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


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
