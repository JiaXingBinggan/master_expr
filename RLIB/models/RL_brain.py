import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import config

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
                 field_nums,
                 action_nums,
                 latent_dims):
        super(Net, self).__init__()
        self.field_nums = field_nums
        self.action_nums = action_nums
        self.latent_dims = latent_dims

        self.layers = list()
        neuron_nums = [200, 300, 100]

        deep_input_dims = self.field_nums * (self.field_nums - 1) // 2 + self.field_nums * self.latent_dims + 3 # b,t,pctr,auc vectors
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
        actions_value = self.mlp(input)
        return actions_value

class PolicyGradient:
    def __init__(
            self,
            action_nums,
            field_nums,
            latent_dims,
            weight_decay=1e-5,
            learning_rate=0.01,
            reward_decay=1,
            device='cuda:0',
    ):
        self.action_nums = action_nums
        self.field_nums = field_nums
        self.latent_dims = latent_dims
        self.lr = learning_rate
        self.gamma = reward_decay
        self.weight_decay = weight_decay
        self.device = device

        self.ep_states, self.ep_as, self.ep_rs = [], [], [] # 状态，动作，奖励，在一轮训练后存储

        self.policy_net = Net(self.field_nums, self.action_nums, self.latent_dims).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def loss_func(self, all_act_prob, acts, vt):
        neg_log_prob = torch.sum(-torch.log(all_act_prob.gather(1, acts - 1))).to(self.device)
        loss = torch.mean(torch.mul(neg_log_prob, vt)).to(self.device)
        return loss

    # 依据概率来选择动作，本身具有随机性
    def choose_action(self, state):
        self.policy_net.train()
        torch.cuda.empty_cache()
        prob_weights = torch.softmax(self.policy_net.forward(state), dim=-1).cpu().detach().numpy()
        action = np.random.choice(range(1, self.action_nums + 1), p=prob_weights.ravel())
        return action

    def choose_best_action(self, state):
        self.policy_net.eval()
        return torch.max(self.policy_net.forward(state), 1)[1].item()

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

        # 对每一回合的奖励，进行折扣计算以及归一化
        discounted_ep_rs_norm = self.discount_and_norm_rewards()
        states = torch.FloatTensor(self.ep_states).to(self.device)
        acts = torch.unsqueeze(torch.LongTensor(self.ep_as), 1).to(self.device)
        vt = torch.FloatTensor(discounted_ep_rs_norm).to(self.device)

        all_act_probs = self.policy_net.forward(states).squeeze(1)

        loss = self.loss_func(all_act_probs, acts, vt)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 训练完后清除训练数据，开始下一轮
        self.ep_states, self.ep_as, self.ep_rs = [], [], []
        return discounted_ep_rs_norm

    # 只存储获得最优收益（点击）那一轮的参数
    def para_store_iter(self, test_results):
        max = 0
        if len(test_results) >= 3:
            for i in range(len(test_results)):
                if i == 0:
                    max = test_results[i]
                elif i != len(test_results) - 1:
                    if test_results[i] > test_results[i - 1] and test_results[i] > test_results[i + 1]:
                        if max < test_results[i]:
                            max = test_results[i]
                else:
                    if test_results[i] > max:
                        max = test_results[i]
        return max