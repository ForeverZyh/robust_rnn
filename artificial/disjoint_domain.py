import itertools

from artificial.AIs import Points, Boxes
import copy
import numpy as np
import heapq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import bisect


def cuboid_data(o, size=(1, 1, 1)):
    # code taken from
    # https://stackoverflow.com/a/35978146/4124317
    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],
         [o[1], o[1], o[1], o[1], o[1]],
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]
    z = [[o[2], o[2], o[2], o[2], o[2]],
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]
    return np.array(x), np.array(y), np.array(z)


def plotCubeAt(pos=(0, 0, 0), size=(1, 1, 1), ax=None, **kwargs):
    # Plotting a cube element at position pos
    if ax != None:
        X, Y, Z = cuboid_data(pos, size)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, **kwargs)


def log_ex4(a, b, c, d):
    # e^a - e^b - e^c + e^d, d < a, d < b, d < c
    if not (np.exp(a - d) - np.exp(b - d) - np.exp(c - d) + 1 > 1e-7):
        return d - 18
    return d + np.log1p(np.exp(a - d) - np.exp(b - d) - np.exp(c - d))


def log_ex3(a, b, c):
    # e^a - e^b - e^c + e^d
    if b < c:
        b, c = c, b
    # c < b, c < a
    return c + np.log(np.exp(a - c) - np.exp(b - c) - 1)


def log_ex2(a, b):
    # e^a - e^b , a > b
    return b + np.log(np.exp(a - b) - 1)


class Disjoint_Domain:
    def __init__(self, budget, points=None, boxes=None):
        self.points = Points([] if points is None else points)
        self.boxes = Boxes([] if boxes is None else boxes)
        self.budget = budget

    def get_volume(self):
        if len(self.boxes.boxes) == 0:
            return 0
        px = []
        vis = set()
        for i in range(len(self.boxes.boxes[0][0])):
            px.append([])
            for b in self.boxes.boxes:
                px[-1].append(b[0][i] - b[1][i])
                px[-1].append(b[0][i] + b[1][i])
            px[-1] = np.unique(px[-1])
            px[-1].sort()

        volume = 0
        for b in self.boxes.boxes:
            iter = ()
            for i in range(len(b[0])):
                lower = bisect.bisect_left(px[i], b[0][i] - b[1][i])
                upper = bisect.bisect_left(px[i], b[0][i] + b[1][i])
                iter += (range(lower, upper),)
            for i in itertools.product(*iter):
                if i not in vis:
                    v = 1
                    for j in range(len(i)):
                        v *= px[j][i[j] + 1] - px[j][i[j]]
                    volume += v
                    vis.add(i)
        return volume

    def get_robust_label(self, nb_classes):
        for i in range(nb_classes):  # two
            flag = True
            for p in self.points.points:
                for j in range(len(p)):
                    if i != j and p[j] >= p[i]:
                        flag = False
                        break
                if not flag:
                    break
            if flag:
                for b in self.boxes.boxes:
                    for j in range(len(b[0])):
                        if i != j and b[0][j] + b[1][j] >= b[0][i] - b[1][i]:
                            flag = False
                            break
                    if not flag:
                        break
            if flag:
                return i
        return -1

    def __len__(self):
        return len(self.points) + len(self.boxes)

    def join(self, other):
        self.points.extend(other.points)
        self.boxes.extend(other.boxes)
        self.check_balance()

    def check_balance(self):
        if len(self.boxes) + len(self.points) <= self.budget:
            return

        # discard similar points and boxes
        self.points.eliminate_similar()
        self.boxes.eliminate_similar()
        for b in self.boxes:
            self.points.eliminate_within_box(b)

        # merge them
        t = len(self.boxes) + len(self.points) - self.budget
        if t <= 0:
            return

        # self.print()
        heap = []
        for i in range(len(self.points.points)):
            for j in range(i + 1, len(self.points.points)):
                cost = np.sum(np.log(np.abs(self.points.points[i] - self.points.points[j])))
                heapq.heappush(heap, (cost, ("PP", i, j)))

            for j in range(len(self.boxes.boxes)):
                cost = log_ex2(np.sum(np.log(
                    2 * np.maximum(np.abs(self.boxes.boxes[j][0] - self.points.points[i]), self.boxes.boxes[j][1]))),
                    np.sum(np.log(2 * self.boxes.boxes[j][1])))
                # cost = np.sum(
                #     np.maximum(np.abs(self.boxes.boxes[j][0] - self.points.points[i]) - self.boxes.boxes[j][1], 0))
                heapq.heappush(heap, (cost, ("PB", i, j)))

        for i in range(len(self.boxes.boxes)):
            for j in range(i + 1, len(self.boxes.boxes)):
                t1 = (self.boxes.boxes[i][0] - self.boxes.boxes[i][1],
                      self.boxes.boxes[j][0] - self.boxes.boxes[j][1])
                t2 = (self.boxes.boxes[i][0] + self.boxes.boxes[i][1],
                      self.boxes.boxes[j][0] + self.boxes.boxes[j][1])
                lower = np.minimum(*t1)
                upper = np.maximum(*t2)
                inter_lower = np.maximum(*t1)
                inter_upper = np.minimum(*t2)
                a = np.sum(np.log(upper - lower))
                b = np.sum(np.log(self.boxes.boxes[i][1] * 2))
                c = np.sum(np.log(self.boxes.boxes[j][1] * 2))
                if (np.any(inter_lower >= inter_upper)):
                    # log_ex3
                    cost = log_ex3(a, b, c)
                else:
                    d = np.sum(np.log(inter_upper - inter_lower))
                    cost = log_ex4(a, b, c, d)
                # cost = np.sum((upper - lower) / 2 - self.boxes.boxes[i][1] - self.boxes.boxes[j][1] + np.maximum(
                #     inter_upper - inter_lower, 0) / 2)
                # cost = np.sum((upper - lower) / 2 - self.boxes.boxes[i][1] - self.boxes.boxes[j][1])
                heapq.heappush(heap, (cost, ("BB", i, j)))

        delete_points = set()
        delete_boxes = set()
        print(t, len(self.points) + len(self.boxes))
        ii = 0
        while ii <= t:
            _, top = heapq.heappop(heap)
            if top[0] == "PP":
                if top[1] in delete_points or top[2] in delete_points:
                    continue
                delete_points.add(top[1])
                delete_points.add(top[2])
                self.boxes.boxes.append([(self.points.points[top[1]] + self.points.points[top[2]]) / 2,
                                         np.abs(self.points.points[top[1]] - self.points.points[top[2]]) / 2])
            elif top[0] == "PB":
                if top[1] in delete_points or top[2] in delete_boxes:
                    continue
                delete_points.add(top[1])
                delete_boxes.add(top[2])
                lower = np.minimum(self.boxes.boxes[top[2]][0] - self.boxes.boxes[top[2]][1],
                                   self.points.points[top[1]])
                upper = np.maximum(self.boxes.boxes[top[2]][0] + self.boxes.boxes[top[2]][1],
                                   self.points.points[top[1]])
                self.boxes.boxes.append([(lower + upper) / 2, (upper - lower) / 2])
            elif top[0] == "BB":
                if top[1] in delete_boxes or top[2] in delete_boxes:
                    continue
                delete_boxes.add(top[1])
                delete_boxes.add(top[2])
                lower = np.minimum(self.boxes.boxes[top[2]][0] - self.boxes.boxes[top[2]][1],
                                   self.boxes.boxes[top[1]][0] - self.boxes.boxes[top[1]][1])
                upper = np.maximum(self.boxes.boxes[top[2]][0] + self.boxes.boxes[top[2]][1],
                                   self.boxes.boxes[top[1]][0] + self.boxes.boxes[top[1]][1])
                self.boxes.boxes.append([(lower + upper) / 2, (upper - lower) / 2])
            ii += 1
            if ii == t:
                break
            for i in range(len(self.points.points)):
                if i not in delete_points:
                    # cost = np.sum(
                    #     np.maximum(np.abs(self.boxes.boxes[-1][0] - self.points.points[i]) - self.boxes.boxes[-1][1],
                    #                0))
                    cost = log_ex2(np.sum(np.log(
                        2 * np.maximum(np.abs(self.boxes.boxes[-1][0] - self.points.points[i]),
                                       self.boxes.boxes[-1][1]))),
                        np.sum(np.log(2 * self.boxes.boxes[-1][1])))
                    heapq.heappush(heap, (cost, ("PB", i, len(self.boxes) - 1)))
            for i in range(len(self.boxes.boxes) - 1):
                if i not in delete_boxes:
                    t1 = (self.boxes.boxes[i][0] - self.boxes.boxes[i][1],
                          self.boxes.boxes[-1][0] - self.boxes.boxes[-1][1])
                    t2 = (self.boxes.boxes[i][0] + self.boxes.boxes[i][1],
                          self.boxes.boxes[-1][0] + self.boxes.boxes[-1][1])
                    lower = np.minimum(*t1)
                    upper = np.maximum(*t2)
                    inter_lower = np.maximum(*t1)
                    inter_upper = np.minimum(*t2)
                    a = np.sum(np.log(upper - lower))
                    b = np.sum(np.log(self.boxes.boxes[i][1] * 2))
                    c = np.sum(np.log(self.boxes.boxes[-1][1] * 2))
                    if (np.any(inter_lower >= inter_upper)):
                        # log_ex3
                        cost = log_ex3(a, b, c)
                    else:
                        d = np.sum(np.log(inter_upper - inter_lower))
                        cost = log_ex4(a, b, c, d)
                    # cost = np.sum((upper - lower) / 2 - self.boxes.boxes[i][1] - self.boxes.boxes[-1][1] + np.maximum(
                    #     inter_upper - inter_lower, 0) / 2)
                    # cost = np.sum((upper - lower) / 2 - self.boxes.boxes[i][1] - self.boxes.boxes[-1][1])
                    heapq.heappush(heap, (cost, ("BB", i, len(self.boxes) - 1)))

        self_points = self.points.points
        self.points.points = []
        for i in range(len(self_points)):
            if i not in delete_points:
                self.points.points.append(self_points[i])

        self_boxes = self.boxes.boxes
        self.boxes.boxes = []
        for i in range(len(self_boxes)):
            if i not in delete_boxes:
                self.boxes.boxes.append(self_boxes[i])
        # self.print()

    def matmul(self, W):
        self.points.matmul(W)
        self.boxes.matmul(W)

    def bias_add(self, b):
        self.points.bias_add(b)
        self.boxes.bias_add(b)

    def __getitem__(self, item):
        return Disjoint_Domain(self.budget, self.points[item], self.boxes[item])

    def __add__(self, other):
        ret = Disjoint_Domain(self.budget, copy.deepcopy(self.points.points), copy.deepcopy(self.boxes.boxes))
        ret.points.cartesian_add(other.points)
        new_other_box = copy.deepcopy(other.boxes)
        # print(other.boxes.boxes)
        new_other_box.cartesian_add(self.points)
        ret.boxes.cartesian_add(other)
        ret.boxes.extend(new_other_box)
        ret.check_balance()
        return ret

    def __mul__(self, other):
        ret = Disjoint_Domain(self.budget, copy.deepcopy(self.points.points), copy.deepcopy(self.boxes.boxes))
        ret.points.cartesian_mul(other.points)
        new_other_box = copy.deepcopy(other.boxes)
        new_other_box.cartesian_mul(self.points)
        ret.boxes.cartesian_mul(other)
        ret.boxes.extend(new_other_box)
        ret.check_balance()
        return ret

    def activation(self, act):
        self.points.activation(act)
        self.boxes.activation(act)

    def GRU_merge(self, self_act, a, b, act):
        # self_act(self) * a + (1- self_act(self)) * act(b)
        new_self_box = self.points.GRU_merge(self_act, a, b, act)
        self.boxes.GRU_merge(self_act, a, b, act)
        self.boxes.boxes += new_self_box
        self.check_balance()

    def print(self):
        print("#points: %d", len(self.points))
        print("#boxes: %d", len(self.boxes))
        try:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            X = []
            Y = []
            Z = []
            x_lower = 1e20
            x_size = 0
            y_lower = 1e20
            y_size = 0
            z_lower = 1e20
            z_size = 0
            for p in self.points.points:
                X.append(p[0])
                x_lower = min(x_lower, p[0])
                x_size = max(x_size, p[0] - x_lower)
                Y.append(p[1])
                y_lower = min(y_lower, p[1])
                y_size = max(y_size, p[1] - y_lower)
                Z.append(p[2])
                z_lower = min(z_lower, p[2])
                z_size = max(z_size, p[2] - z_lower)
            ax.scatter(X, Y, Z, marker='o')
            X = []
            Y = []
            Z = []

            for x in self.boxes.boxes:
                if np.sum(x[1]) > 1e-2:
                    x_lower = min(x_lower, x[0][0] - x[1][0])
                    y_lower = min(y_lower, x[0][1] - x[1][1])
                    z_lower = min(z_lower, x[0][2] - x[1][2])
                    x_size = max(x_size, x[1][0] + x[0][0] - x_lower)
                    y_size = max(y_size, x[1][1] + x[0][1] - y_lower)
                    z_size = max(z_size, x[1][2] + x[0][2] - z_lower)
                    plotCubeAt(pos=x[0] - x[1], size=x[1] * 2, ax=ax, color="b")
                else:
                    X.append(x[0][0])
                    x_lower = min(x_lower, x[0][0])
                    x_size = max(x_size, x[0][0] - x_lower)
                    Y.append(x[0][1])
                    y_lower = min(y_lower, x[0][1])
                    y_size = max(y_size, x[0][1] - y_lower)
                    Z.append(x[0][2])
                    z_lower = min(z_lower, x[0][2])
                    z_size = max(z_size, x[0][2] - z_lower)
            ax.scatter(X, Y, Z, marker='^')

            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            x_size = x_size * 1.5 if x_size > 0 else 1
            y_size = y_size * 1.5 if y_size > 0 else 1
            z_size = z_size * 1.5 if z_size > 0 else 1
            ax.set_xlim([x_lower - x_size / 2, x_lower + x_size])
            ax.set_ylim([y_lower - y_size / 2, y_lower + y_size])
            ax.set_zlim([z_lower - z_size / 2, z_lower + z_size])
            plt.show()
        except:
            pass
        for p in self.points:
            print(p)

        for b in self.boxes:
            print(b)
