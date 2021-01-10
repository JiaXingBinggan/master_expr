# import torch
# import torch.nn as nn
# from torch.distributions import MultivariateNormal, Categorical, Normal
#
# x = torch.softmax(torch.Tensor([[0.1, 0.2, 0.7],[0.3, 0.4, 0.2]]), dim=-1)
# print(torch.softmax(x, dim=-1))
# print(torch.softmax(x, dim=1))
#
# print(x)
# y = torch.exp(torch.log(x))
# print(y)
#
# t = torch.diag(torch.full((3,), 1*1))
# print(t)
#
# o = 0.9
# for i in range(300):
#     if (i + 1) % 10 == 0:
#         o -= 10/300
#         print(o)
#
# u = MultivariateNormal(x, t)
# k = u.sample()
# p = u.log_prob(k)
# print(p)
# print(k)
# print(torch.softmax(k, dim=1))
#
# t = torch.full((3,), 1*1).expand_as(x)
# print(torch.diag_embed(t))
#
# C = Categorical(torch.softmax(x, dim=1))
# print(C.sample() + 2)
#
# class Hybrid_Actor_Critic(nn.Module):
#     def __init__(self, input_dims, action_nums):
#         super(Hybrid_Actor_Critic, self).__init__()
#         self.input_dims = input_dims
#
#         neuron_nums = [300, 300, 300]
#
#         # Critic
#         self.Critic = nn.Sequential(
#             nn.Linear(self.input_dims, neuron_nums[0]),
#             nn.BatchNorm1d(neuron_nums[0]),
#             nn.ReLU(),
#             nn.Linear(neuron_nums[0], neuron_nums[1]),
#             nn.BatchNorm1d(neuron_nums[1]),
#             nn.ReLU(),
#             nn.Linear(neuron_nums[1], neuron_nums[2]),
#             nn.BatchNorm1d(neuron_nums[2]),
#             nn.ReLU(),
#             nn.Linear(neuron_nums[2], 1)
#         )
#
#         # Continuous_Actor
#         self.Continuous_Actor = nn.Sequential(
#             nn.Linear(self.input_dims, neuron_nums[0]),
#             nn.BatchNorm1d(neuron_nums[0]),
#             nn.ReLU(),
#             nn.Linear(neuron_nums[0], neuron_nums[1]),
#             nn.BatchNorm1d(neuron_nums[1]),
#             nn.ReLU(),
#             nn.Linear(neuron_nums[1], neuron_nums[2]),
#             nn.BatchNorm1d(neuron_nums[2]),
#             nn.ReLU(),
#             nn.Linear(neuron_nums[2], action_nums)
#         )
#
#         # Discrete_Actor
#         self.Discrete_Actor = nn.Sequential(
#             nn.Linear(self.input_dims, neuron_nums[0]),
#             nn.BatchNorm1d(neuron_nums[0]),
#             nn.ReLU(),
#             nn.Linear(neuron_nums[0], neuron_nums[1]),
#             nn.BatchNorm1d(neuron_nums[1]),
#             nn.ReLU(),
#             nn.Linear(neuron_nums[1], neuron_nums[2]),
#             nn.BatchNorm1d(neuron_nums[2]),
#             nn.ReLU(),
#             nn.Linear(neuron_nums[2], action_nums)
#         )
#
#
#
# Hybrid_Actor_Critic(10,2)
#
# m = Normal(torch.tensor([0.0, 1.0, 2.0]), torch.tensor([1.0]))
# k = m.sample()
# o = m.sample()
# l = m.log_prob(k)
# l1 = m.log_prob(o)
# print(k, l, o)
#
# print(torch.mean((l - l1).exp()))
#
# init_lr = 1e-2
# end_lr = 1e-4
# lr_lamda = (init_lr - end_lr) / 400
#
# for i in range(500):
#     print(max(init_lr - lr_lamda * (i + 1), end_lr))
#
# beta = torch.min(torch.FloatTensor([1., 0.5 + 0.01]))
# print(beta)
#
# print(torch.min)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ffm_preds = pd.read_csv('../../data/avazu/avazu/IPNN/test_submission.csv', header=None).values
fm_preds = pd.read_csv('../../data/avazu/avazu/AFM/test_submission.csv', header=None).values
lr_preds = pd.read_csv('../../data/avazu/avazu/DCN/test_submission.csv', header=None).values
test_data = pd.read_csv('../../data/avazu/avazu/test_.txt', header=None).values

with_clk_indexs = np.where(test_data[:, 0] == 1)
ffm_preds_clk = ffm_preds[with_clk_indexs][:, 1]
fm_preds_clk = fm_preds[with_clk_indexs][:, 1]
lr_preds_clk = lr_preds[with_clk_indexs][:, 1]
print(len(ffm_preds_clk), len(np.where(ffm_preds_clk >= fm_preds_clk)[0]), len(np.where(ffm_preds_clk >= lr_preds_clk)[0]), len(np.where(fm_preds_clk >= lr_preds_clk)[0]))
print(len(np.where(ffm_preds_clk >= (ffm_preds_clk + fm_preds_clk + lr_preds_clk) / 3)[0]))

plt.plot(with_clk_indexs[0], ffm_preds_clk, 'r', label='ffm')
# plt.plot(with_clk_indexs[0], fm_preds_clk, 'b', label='fm')
# plt.plot(with_clk_indexs[0], lr_preds_clk, 'g', label='lr')
plt.plot(with_clk_indexs[0], (ffm_preds_clk + fm_preds_clk + lr_preds_clk) / 3, 'y', label='avg')

plt.legend()
plt.show()

without_clk_indexs = np.where(test_data[:, 0] == 0)
ffm_preds_withoutclk = ffm_preds[without_clk_indexs][:, 1]
fm_preds_withoutclk = fm_preds[without_clk_indexs][:, 1]
lr_preds_withoutclk = lr_preds[without_clk_indexs][:, 1]
print(len(ffm_preds_withoutclk), len(np.where(ffm_preds_withoutclk <= fm_preds_withoutclk)[0]), len(np.where(ffm_preds_withoutclk <= lr_preds_withoutclk)[0]), len(np.where(fm_preds_withoutclk <= lr_preds_withoutclk)[0]))
print(len(np.where(ffm_preds_withoutclk <= (ffm_preds_withoutclk + fm_preds_withoutclk + lr_preds_withoutclk) / 3)[0]))

plt.plot(without_clk_indexs[0], ffm_preds_withoutclk, 'r', label='ffm')
# plt.plot(without_clk_indexs[0], fm_preds_withoutclk, 'b', label='fm')
# plt.plot(without_clk_indexs[0], lr_preds_withoutclk, 'g', label='lr')
plt.plot(without_clk_indexs[0], (ffm_preds_withoutclk + fm_preds_withoutclk + lr_preds_withoutclk) / 3, 'y', label='avg')

plt.legend()
plt.show()