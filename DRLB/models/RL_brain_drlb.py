import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
from torch.autograd import Variable

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
    def __init__(self, feature_dims, action_nums, neuron_nums):
        super(Net, self).__init__()
        self.neuron_nums = neuron_nums
        self.action_nums = action_nums
        
        deep_input_dims = feature_dims
        self.bn_input = nn.BatchNorm1d(deep_input_dims)
        self.bn_input.weight.data.fill_(1)
        self.bn_input.bias.data.fill_(0)

        self.layers = list()
        for neuron_num in self.neuron_nums:
            self.layers.append(nn.Linear(deep_input_dims, neuron_num))
            self.layers.append(nn.ReLU())
            deep_input_dims = neuron_num

        self.layers.append(nn.Linear(deep_input_dims, self.action_nums))

        weight_init([self.layers[-1]])

        self.mlp = nn.Sequential(*self.layers)

    def forward(self, input):
        actions_value = self.mlp(self.bn_input(input))
        return actions_value

# 定义DeepQNetwork
class DRLB:
    def __init__(
            self,
            neuron_nums,
            action_dims,  # 动作的数量
            state_dims,  # 状态的特征数量
            lr=0.01,  # 学习率
            reward_decay=1,  # 奖励折扣因子,偶发过程为1
            e_greedy=0.9,  # 贪心算法ε
            replace_target_iter=300,  # 每300步替换一次target_net的参数
            memory_size=500,  # 经验池的大小
            batch_size=32,  # 每次更新时从memory里面取多少数据出来，mini-batch
            device='cuda: 0',
    ):
        self.neuron_nums = neuron_nums
        self.action_dims = action_dims  # 动作的具体数值？[0,0.01,...,budget]
        self.state_dims = state_dims
        self.lr = lr
        self.gamma = reward_decay
        self.epsilon_max = e_greedy  # epsilon 的最大值
        self.replace_target_iter = replace_target_iter  # 更换 target_net 的步数
        self.memory_size = memory_size  # 记忆上限
        self.batch_size = batch_size  # 每次更新时从 memory 里面取多少记忆出来
        self.epsilon = 0.9
        self.device = device

        # 设置随机数种子
        setup_seed(1)

        # hasattr(object, name)
        # 判断一个对象里面是否有name属性或者name方法，返回BOOL值，有name特性返回True， 否则返回False。
        # 需要注意的是name要用括号括起来
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        # 记录学习次数（用于判断是否替换target_net参数）
        self.learn_step_counter = 0

        # 将经验池<状态-动作-奖励-下一状态>中的转换组初始化为0
        self.memory = torch.zeros((self.memory_size, self.state_dims * 2 + 3))  # 状态的特征数*2加上动作和奖励

        # 创建target_net（目标神经网络），eval_net（训练神经网络）
        self.eval_net = Net(self.state_dims, self.action_dims, self.neuron_nums).to(self.device)
        self.target_net = Net(self.state_dims, self.action_dims, self.neuron_nums).to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())

        # 优化器
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)

    # 经验池存储，s-state, a-action, r-reward, s_-state_, done
    def store_transition(self, transition):
        # 由于已经定义了经验池的memory_size，如果超过此大小，旧的memory则被新的memory替换
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition  # 替换
        self.memory_counter += 1

    # 重置epsilon
    def reset_epsilon(self, e_greedy):
        self.epsilon = e_greedy

    def up_learn_step(self):
        self.learn_step_counter += 1

    # 选择动作
    def choose_action(self, state):
        torch.cuda.empty_cache()

        random_probability = max(self.epsilon, 0.5) # 论文的取法
        self.eval_net.eval()
        with torch.no_grad():
            if np.random.uniform() > random_probability:
                action = torch.argmax(self.eval_net.forward(state), dim=-1)
            else:
                action = np.random.randint(0, self.action_dims)

        return action

    # 选择最优动作
    def choose_best_action(self, state):
        self.eval_net.eval()
        with torch.no_grad():
            action = torch.argmax(self.eval_net.forward(state), dim=-1)
        return action

    # 定义DQN的学习过程
    def learn(self):
        # 清除显存缓存
        torch.cuda.empty_cache()

        # 检查是否达到了替换target_net参数的步数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            # print(('\n目标网络参数已经更新\n'))
        self.learn_step_counter += 1

        # 训练过程
        # 从memory中随机抽取batch_size的数据
        if self.memory_counter > self.memory_size:
            # replacement 代表的意思是抽样之后还放不放回去，如果是False的话，那么出来的三个数都不一样，如果是True的话， 有可能会出现重复的，因为前面的抽的放回去了
            sample_index = random.sample(range(self.memory_size), self.batch_size)
        else:
            sample_index = random.sample(range(self.memory_counter), self.batch_size)

        batch_memory = self.memory[sample_index, :].to(self.device)

        # 获取到q_next（target_net产生）以及q_eval（eval_net产生）
        # 如store_transition函数中存储所示，state存储在[0, feature_dims-1]的位置（即前feature_numbets）
        # state_存储在[feature_dims+1，memory_size]（即后feature_dims的位置）
        b_s = batch_memory[:, :self.state_dims]
        b_a = batch_memory[:, self.state_dims].unsqueeze(1).long()
        b_r = batch_memory[:, self.state_dims + 1]
        b_s_ = batch_memory[:, -self.state_dims - 1: -1]
        b_done = batch_memory[:, -1].unsqueeze(1)

        # q_eval w.r.t the action in experience
        # b_a - 1的原因是，出价动作最高300，而数组的最大index为299
        q_eval = self.eval_net.forward(b_s).gather(1, b_a)  # shape (batch,1), gather函数将对应action的Q值提取出来做Bellman公式迭代
        q_next = self.target_net.forward(b_s_).detach()  # detach from graph, don't backpropagate，因为target网络不需要训练

        q_target = b_r.view(self.batch_size, 1) + self.gamma * torch.mul(q_next.max(1)[0].view(self.batch_size,
                                                                                     1), 1 - b_done)
        # 训练eval_net
        loss = F.mse_loss(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=40, norm_type=2)
        self.optimizer.step()

        return loss.item()

    def control_epsilon(self, t):
        # 逐渐增加epsilon，增加行为的利用性
        r_epsilon = 2e-5  # 降低速率
        self.epsilon = max(0.95 - r_epsilon * t, 0.05)

