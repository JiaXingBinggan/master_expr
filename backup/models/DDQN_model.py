import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import datetime

np.seterr(all='raise')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(1)

class Net(nn.Module):
    def __init__(self, field_nums, feature_nums, latent_dims, action_nums):
        super(Net, self).__init__()
        self.field_nums = field_nums
        self.feature_nums = feature_nums
        self.latent_dims = latent_dims

        self.input_dims = self.field_nums * (self.field_nums - 1) // 2 + self.field_nums * self.latent_dims

        # self.bn_input = nn.BatchNorm1d(self.input_dims)
        # nn.init.uniform_(self.bn_input,)

        deep_input_dims = self.input_dims
        layers = list()
        neuron_nums = [300, 300, 300]
        for neuron_num in neuron_nums:
            layers.append(nn.Linear(deep_input_dims, neuron_num))
            layers.append(nn.BatchNorm1d(neuron_num))
            layers.append(nn.ReLU())
            deep_input_dims = neuron_num
        #
        # for i, layer in enumerate(layers):
        #     if i % 3 == 0:
        #         nn.init.xavier_uniform_(layer.weight)

        layers.append(nn.Linear(deep_input_dims, action_nums))

        self.mlp = nn.Sequential(*layers)

    def forward(self, input):
        actions_value = self.mlp(input)

        return actions_value

# 定义Double DeepQNetwork
class DoubleDQN:
    def __init__(
            self,
            feature_nums,  # 状态的特征数量
            field_nums,
            latent_dims,
            campaign_id='1458',
            action_nums=3,  # 动作的数量
            learning_rate=1e-3,  # 学习率
            reward_decay=1,  # 奖励折扣因子,偶发过程为1
            replace_target_iter=30,  # 每30次训练则替换一次target_net的参数
            memory_size=300,  # 经验池的大小
            batch_size=32,  # 每次更新时从memory里面取多少数据出来，mini-batch
            device='cuda:0',
    ):
        self.action_nums = action_nums - 1  # 动作的具体数值？[0,0.01,...,budget]
        self.feature_nums = feature_nums
        self.field_nums = field_nums
        self.latent_dims = latent_dims
        self.lr = learning_rate
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter  # 更换 target_net 的步数
        self.memory_size = memory_size  # 记忆上限
        self.batch_size = batch_size  # 每次更新时从 memory 里面取多少记忆出来
        self.device = device
        self.campaign_id = campaign_id

        # hasattr(object, name)
        # 判断一个对象里面是否有name属性或者name方法，返回BOOL值，有name特性返回True， 否则返回False。
        # 需要注意的是name要用括号括起来
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # 记录学习次数（用于判断是否替换target_net参数）
        self.learn_step_counter = 0

        # 将经验池<状态-动作-奖励-下一状态>中的转换组初始化为0
        self.memory = torch.zeros(size=[self.memory_size, self.field_nums + 2]).to(self.device)

        # 创建target_net（目标神经网络），eval_net（训练神经网络）
        self.eval_net, self.target_net = Net(self.field_nums, self.feature_nums, self.latent_dims, self.action_nums).to(self.device), Net(
            self.field_nums, self.feature_nums, self.latent_dims, self.action_nums).to(self.device)

        # 优化器
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr, weight_decay=1e-5)

        # 损失函数为，均方损失函数
        self.loss_func = nn.MSELoss()

    # 经验池存储，s-state, a-action, r-reward
    def store_transition(self, transitions):
        transition_lens = len(transitions)

        # 由于已经定义了经验池的memory_size，如果超过此大小，旧的memory则被新的memory替换
        index_start = self.memory_counter % self.memory_size
        index_end = (self.memory_counter + transition_lens) % self.memory_size

        if index_end > index_start:
            self.memory[index_start: index_end, :] = transitions  # 替换
        else:
            replace_len_1 = self.memory_size - index_start
            self.memory[index_start: self.memory_size, :] = transitions[0: replace_len_1]
            replace_len_2 = transition_lens - replace_len_1
            self.memory[0: replace_len_2, :] = transitions[replace_len_1: transition_lens]

        self.memory_counter += transition_lens

    # def paramter_noise(self, new_eval_net, exploration_rate):
    #     new_eval_net.bn_input.weight.data += torch.normal(0, exploration_rate, size=new_eval_net.bn_input.weight.data.size()).to(self.device)
    #     for i, layer in enumerate(new_eval_net.mlp):
    #         if i % 3 == 0 or (i - 1) % 3 == 0:
    #             layer.weight.data += torch.normal(0, exploration_rate, size=layer.weight.data.size()).to(self.device)
    #
    #     return new_eval_net
    # def choose_action(self, states, exploration_rate):
    #     torch.cuda.empty_cache()
    #
    #     # states = self.embedding_layer.forward(states)
    #     new_eval_net = self.paramter_noise(copy.deepcopy(self.eval_net), exploration_rate)
    #
    #     new_eval_net.eval()
    #     with torch.no_grad():
    #         action_values = new_eval_net.forward(states)
    #
    #     max_action = torch.argsort(-action_values)[:, 0] + 2
    #
    #     # 用矩阵来初始
    #     return max_action.view(-1, 1)
    # 选择动作
    def choose_action(self, states, exploration_rate):
        torch.cuda.empty_cache()

        # states = self.embedding_layer.forward(states)
        self.eval_net.eval()
        with torch.no_grad():
            action_values = self.eval_net.forward(states)
        self.eval_net.train()

        random_seeds = torch.rand(len(states), 1).to(self.device)
        max_action = torch.argsort(-action_values)[:, 0] + 2
        random_action = torch.randint(low=2, high=self.action_nums + 2, size=[len(states), 1]).to(self.device)

        # exploration_rate = max(exploration_rate, 0.1)
        actions = torch.where(random_seeds >= exploration_rate, max_action.view(-1, 1), random_action)

        # 用矩阵来初始
        return actions

    # 选择最优动作
    def choose_best_action(self, states):
        # states = self.embedding_layer.forward(states)

        self.eval_net.eval()
        with torch.no_grad():
            action_values = self.eval_net.forward(states)
            action = torch.argsort(-action_values)[:, 0] + 2

        return action.view(-1, 1)

    def soft_update(self, net, net_target):
        for param_target, param in zip(net_target.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - 0.001) + param.data * 0.001)

    def sample_batch(self):
        # 训练过程
        # 从memory中随机抽取batch_size的数据
        if self.memory_counter > self.memory_size:
            # replacement 代表的意思是抽样之后还放不放回去，如果是False的话，那么出来的三个数都不一样，如果是True的话， 有可能会出现重复的，因为前面的抽的放回去了
            sample_index = torch.LongTensor(random.sample(range(self.memory_size), self.batch_size)).to(self.device)
        else:
            sample_index = torch.LongTensor(random.sample(range(self.memory_counter), self.batch_size)).to(self.device)

        batch_memory = self.memory[sample_index, :].long()

        # 获取到q_next（target_net产生）以及q_eval（eval_net产生）
        b_s = batch_memory[:, :self.field_nums]
        b_a = batch_memory[:, self.field_nums: self.field_nums + 1]
        b_r = batch_memory[:, self.field_nums + 1].view(-1, 1).float()
        b_s_ = batch_memory[:, :self.field_nums]

        return b_s, b_a, b_r, b_s_

    # 定义DQN的学习过程
    def learn(self, b_s, b_a, b_r, b_s_):
        # 清除显存缓存
        torch.cuda.empty_cache()

        # self.soft_update(self.target_net, self.eval_net)
        # # 检查是否达到了替换target_net参数的步数
        if self.learn_step_counter % self.replace_target_iter == 0:
             self.target_net.load_state_dict(self.eval_net.state_dict())
        #     # print(('\n目标网络参数已经更新\n'))
        self.learn_step_counter += 1

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net.forward(b_s).gather(1, b_a - 2)  # shape (batch,1), gather函数将对应action的Q值提取出来做Bellman公式迭代
        q_next = self.target_net.forward(b_s_).detach()  # detach from graph, don't backpropagate，因为target网络不需要训练
        # 下一状态s的eval_net值
        q_eval_next = self.eval_net.forward(b_s_)
        max_b_a_next = torch.unsqueeze(torch.max(q_eval_next, 1)[1], 1)  # 选择最大Q的动作
        select_q_next = q_next.gather(1, max_b_a_next)

        q_target = b_r + self.gamma * select_q_next # shape (batch, 1)

        # 训练eval_net
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()