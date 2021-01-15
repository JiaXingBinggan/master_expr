import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import config
from torch.autograd import Variable

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

class Net(nn.Module):
    def __init__(self,
                 action_nums):
        super(Net, self).__init__()
        self.action_nums = action_nums

        self.layers = list()
        neuron_nums = [64]

        deep_input_dims = 4 # b,t,pctr
        for neuron_num in neuron_nums:
            self.layers.append(nn.Linear(deep_input_dims, neuron_num))
            # self.layers.append(nn.BatchNorm1d(neuron_num))
            self.layers.append(nn.Dropout(p=0.2))
            self.layers.append(nn.ReLU())
            deep_input_dims = neuron_num

        self.layers.append(nn.Linear(deep_input_dims, self.action_nums))

        weight_init(self.layers)

        self.mlp = nn.Sequential(*self.layers)

    def forward(self, input):
        actions_value = gumbel_softmax_sample(self.mlp(input))

        return actions_value

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

class PolicyGradient:
    def __init__(
            self,
            action_nums,
            weight_decay=1e-5,
            learning_rate=0.01,
            reward_decay=1,
            device='cuda:0',
    ):
        self.action_nums = action_nums
        self.lr = learning_rate
        self.gamma = reward_decay
        self.weight_decay = weight_decay
        self.device = device

        self.ep_states, self.ep_as, self.ep_rs = [], [], []

        self.policy_net = Net(self.action_nums).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

    def loss_func(self, all_act_prob, acts, vt):
        log_prob = torch.log(all_act_prob.gather(1, acts)).to(self.device)
        log_probs = torch.mul(log_prob, vt)
        entropies = all_act_prob * all_act_prob.log()
        loss = -(log_probs - entropies).mean()
        return loss

    # 依据概率来选择动作，本身具有随机性
    def choose_action(self, state):
        self.policy_net.train()
        torch.cuda.empty_cache()
        prob_weights = self.policy_net.forward(state).cpu().detach().numpy()

        action = np.random.choice(range(self.action_nums), p=prob_weights.ravel())

        return action

    def choose_best_action(self, state):
        self.policy_net.eval()
        return torch.argmax(self.policy_net.forward(state), dim=-1).item()

    # 储存一回合的s,a,r；因为每回合训练
    def store_transition(self, s, a, r):
        self.ep_states.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    # 对每一轮的奖励值进行累计折扣及归一化处理
    def discount_and_norm_rewards(self):
        discounted_ep_rs = np.zeros_like(self.ep_rs, dtype=np.float)
        running_add = 0
        # reversed 函数返回一个反转的迭代器。
        # 计算折扣后的 reward
        # 公式： E = r1 + r2 * gamma + r3 * gamma * gamma + r4 * gamma * gamma * gamma ...
        for i in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[i]
            discounted_ep_rs[i] = running_add

        # 归一化处理
        discounted_ep_rs -= np.mean(discounted_ep_rs)  # 均值
        discounted_ep_rs /= np.std(discounted_ep_rs)  # 方差
        return discounted_ep_rs

    def learn(self):
        torch.cuda.empty_cache()
        self.policy_net.train()

        # # 对每一回合的奖励，进行折扣计算以及归一化
        discounted_ep_rs_norm = self.discount_and_norm_rewards()
        states = torch.FloatTensor(self.ep_states).to(self.device)
        acts = torch.unsqueeze(torch.LongTensor(self.ep_as), 1).to(self.device)
        vt = torch.FloatTensor(discounted_ep_rs_norm).view(-1, 1).to(self.device)

        all_act_probs = self.policy_net.forward(states).squeeze(1)

        loss = self.loss_func(all_act_probs, acts, vt)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 40, norm_type=2)
        self.optimizer.step()

        # R = torch.zeros(1, 1)
        # loss = 0
        # for i in reversed(range(len(self.ep_rs))):
        #     R = self.gamma * R + self.ep_rs[i]
        #     loss = loss - (self.ep_logps[i] * (Variable(R).expand_as(self.ep_logps[i])).to(self.device)).sum() - (
        #                 0.001 * self.ep_entropies[i].to(self.device)).sum()
        # loss = loss / len(self.ep_rs)
        #
        # self.optimizer.zero_grad()
        # loss.backward()
        # nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10, norm_type=2)
        # self.optimizer.step()

        # 训练完后清除训练数据，开始下一轮
        self.ep_states, self.ep_as, self.ep_rs = [], [], []
        return loss.item()
