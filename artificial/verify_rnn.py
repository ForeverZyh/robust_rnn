import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np
import copy
import matplotlib.pyplot as plt

from simple_rnn_model import params, SimpleRNNModel, SimpleRNNModel1
from utils import get_one_hot
from artificial.disjoint_domain import Disjoint_Domain
from artificial.AIs import Points, Boxes

nb_classes = 2
# params["max_len"] = 100
# model1 = SimpleRNNModel1(params, 30, nb_classes)
# model1.model.load_weights("./models/rnn")
params["max_len"] = 100
params["D"] = 2
params["conv_layer2_nfilters"] = 3
model1 = SimpleRNNModel1(params, 30, nb_classes)
model1.model.load_weights("./models/rnn_tiny")
budget = 5
sub_num = 2
units = params["conv_layer2_nfilters"]
kernel, recurrent_kernel, bias = model1.gru.get_weights()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def GRU(h_tm1: Disjoint_Domain, input: Disjoint_Domain):
    # inputs projected by all gate matrices at once
    input = copy.deepcopy(input)
    matrix_inner = copy.deepcopy(h_tm1)
    input.matmul(kernel)
    input.bias_add(bias)
    x_z = input[:units]
    x_r = input[units: 2 * units]
    x_h = input[2 * units:]

    # hidden state projected separately for update/reset and new
    matrix_inner.matmul(recurrent_kernel[:, :2 * units])

    # print("matrix_inner")
    # matrix_inner.print()
    recurrent_z = matrix_inner[:units]
    recurrent_r = matrix_inner[units: 2 * units]

    # print("xz")
    # x_z.print()
    # print("recurrent_z")
    # recurrent_z.print()
    z = x_z + recurrent_z
    r = x_r + recurrent_r
    r.activation(sigmoid)

    # print("r")
    # r.print()
    # print("h_tm1")
    # h_tm1.print()
    recurrent_h = r * h_tm1
    recurrent_h.matmul(recurrent_kernel[:, 2 * units:])

    hh = x_h + recurrent_h
    # hh.activation(self.activation)

    # previous and candidate state mixed by update gate
    z.GRU_merge(sigmoid, h_tm1, hh, tanh)
    return z


def Box_without_ed(initial_input, modify_input, max_len):
    F = [Disjoint_Domain(budget) for _ in range(max_len + 1)]
    F[0] = Disjoint_Domain(budget, [np.zeros(params["conv_layer2_nfilters"])])
    for i in range(1, 1 + max_len):
        x = copy.deepcopy(initial_input[i - 1])
        x.join(modify_input[i - 1])
        F[i].join(GRU(F[i - 1], x))

    # F[max_len][sub_num].print()
    ans = copy.deepcopy(F[max_len])
    W, b = model1.fc1.get_weights()
    ans.matmul(W)
    ans.bias_add(b)
    # ans.print()
    # print(ans.get_volume())
    return ans.get_robust_label(2), ans.get_volume()
    # F[max_len][sub_num].print()


def DP(initial_input, modify_input, max_len):
    F = [[Disjoint_Domain(budget) for _ in range(sub_num + 1)] for _ in range(max_len + 1)]
    F[0][0] = Disjoint_Domain(budget, [np.zeros(params["conv_layer2_nfilters"])])
    for i in range(1, 1 + max_len):
        # print(i)
        for j in range(sub_num + 1):
            F[i][j].join(GRU(F[i - 1][j], initial_input[i - 1]))
            if j > 0:
                F[i][j].join(GRU(F[i - 1][j - 1], modify_input[i - 1]))
            # F[i][j].print()

    # F[max_len][sub_num].print()
    ans = copy.deepcopy(F[max_len][0])
    for i in range(1, sub_num):
        ans.join(F[max_len][i])
    W, b = model1.fc1.get_weights()
    ans.matmul(W)
    ans.bias_add(b)
    # ans.print()
    # print(ans.get_volume())
    return ans.get_robust_label(2), ans.get_volume()
    # F[max_len][sub_num].print()


np.random.seed(19970402)


def get_valid_input():
    length = 100
    char = 26
    while True:
        t = np.random.randint(1, length + 1)
        x = np.zeros(length, dtype=np.int)
        cnt_a = 0
        for j in range(t):
            x[j] = np.random.randint(1, char + 2)
            if x[j] == 1:
                cnt_a += 1

        if cnt_a > 2 + 2:
            break
    print(x)
    return x


embedding = model1.embed.get_weights()[0]

# for i, p in enumerate(embedding):
#     x = p[0]
#     y = p[1]
#     plt.scatter(x, y, marker='x', color='red')
#     # plt.text(x + 0.1, y + 0.1, chr(i + 96) if i > 0 else ' ', fontsize=9)
# plt.show()

cnt = 0
vol_sum = 0
for _ in range(100):
    x = get_valid_input()
    initial_input = []
    modify_input = []
    for i in range(params["max_len"]):
        initial_input.append(Disjoint_Domain(budget, points=[embedding[x[i]]]))
        if x[i] == 1:
            # modify_input.append(Disjoint_Domain(budget))
            modify_input.append(Disjoint_Domain(budget, points=[embedding[j] for j in range(2, 27)]))
        elif x[i] > 0:
            modify_input.append(Disjoint_Domain(budget))
            # modify_input.append(Disjoint_Domain(budget, points=[embedding[1]]))
        else:
            modify_input.append(Disjoint_Domain(budget))
            # modify_input.append(Disjoint_Domain(budget, points=[embedding[j] for j in range(2, 27)]))
            # modify_input.append(Disjoint_Domain(budget, points=[embedding[0]]))
        modify_input[-1].check_balance()
    ans, vol = DP(initial_input, modify_input, 100)
    # ans, vol = Box_without_ed(initial_input, modify_input, 100)
    cnt += 1 if ans == 1 else 0
    vol_sum += vol

print(cnt / 100.0, vol_sum / 100.0)
