import numpy as np


def GRU_merge(xx, self_act, aa, bb, act):
    if isinstance(xx, list):
        xx = [xx[0] - xx[1], xx[0] + xx[1]]
    else:
        xx = [xx]

    if isinstance(aa, list):
        aa = [aa[0] - aa[1], aa[0] + aa[1]]
    else:
        aa = [aa]

    if isinstance(bb, list):
        bb = [bb[0] - bb[1], bb[0] + bb[1]]
    else:
        bb = [bb]

    res = []
    for x in xx:
        for a in aa:
            for b in bb:
                res.append(self_act(x) * a + (1 - self_act(x)) * act(b))

    if len(res) == 1:
        assert len(res[0].shape) == 1
        return res[0]
    else:
        lower = np.min(res, axis=0)
        upper = np.max(res, axis=0)
        assert len(lower.shape) == 1
        return [(lower + upper) / 2, (upper - lower) / 2]


class Points:
    def __init__(self, points):
        self.points = points

    def extend(self, other):
        self.points += other.points

    def eliminate_similar(self):
        self_points = self.points
        self.points = []
        for i in range(len(self_points)):
            min_dis = 1e10
            for j in range(i + 1, len(self_points)):
                min_dis = min(min_dis, np.sum(np.abs(self_points[i] - self_points[j])))
            if min_dis > 1e-7:
                self.points.append(self_points[i])

    def eliminate_within_box(self, b):
        self_points = self.points
        self.points = []
        for i in range(len(self_points)):
            if not (np.all(np.less_equal(b[0] - b[1], self_points[i])) and np.all(
                    np.less_equal(self_points[i], b[0] + b[1]))):
                self.points.append(self_points[i])

    def cartesian_add(self, other):
        self_points = self.points
        self.points = []
        for x in self_points:
            for y in other.points:
                self.points.append(x + y)

    def cartesian_mul(self, other):
        self_points = self.points
        self.points = []
        for x in self_points:
            for y in other.points:
                self.points.append(x * y)

    def GRU_merge(self, self_act, aa, bb, act):
        self_points = self.points
        self.points = []
        ret_boxes = []
        for x in self_points:
            for a in aa.points:
                for b in bb.points:
                    self.points.append(GRU_merge(x, self_act, a, b, act))
                for b in bb.boxes:
                    ret_boxes.append(GRU_merge(x, self_act, a, b, act))

            for a in aa.boxes:
                for b in bb.points:
                    ret_boxes.append(GRU_merge(x, self_act, a, b, act))
                for b in bb.boxes:
                    ret_boxes.append(GRU_merge(x, self_act, a, b, act))

        return ret_boxes

    def to_Boxes(self):
        assert len(self.points) > 0
        lower = np.min(np.array(self.points), axis=0)
        upper = np.max(np.array(self.points), axis=0)
        return [(lower + upper) / 2, (upper - lower) / 2]

    def __len__(self):
        return len(self.points)

    def matmul(self, W):
        for i in range(len(self.points)):
            self.points[i] = np.matmul(self.points[i], W)

    def bias_add(self, b):
        for i in range(len(self.points)):
            self.points[i] += b

    def activation(self, act):
        for i in range(len(self.points)):
            self.points[i] = act(self.points[i])

    def __getitem__(self, item):
        return [x[item] for x in self.points]

    def __iter__(self):
        return self.points.__iter__()


class Boxes:
    def __init__(self, boxes):
        self.boxes = boxes

    def extend(self, other):
        self.boxes += other.boxes

    def eliminate_similar(self):
        self_boxes = self.boxes
        self.boxes = []
        for i in range(len(self_boxes)):
            flag_del = False
            for j in range(i + 1, len(self_boxes)):
                if (np.all(self_boxes[i][0] - self_boxes[i][1] >= self_boxes[j][0] - self_boxes[j][1] - 1e-7)) and (
                        np.all(self_boxes[i][0] + self_boxes[i][1] <= self_boxes[j][0] + self_boxes[j][1] + 1e-7)):
                    flag_del = True
                    break
            if not flag_del:
                self.boxes.append(self_boxes[i])

    def cartesian_add(self, other):
        self_boxes = self.boxes
        self.boxes = []
        if not isinstance(other, Points):
            for x in self_boxes:
                for y in other.boxes:
                    self.boxes.append([x[0] + y[0], x[1] + y[1]])

        for x in self_boxes:
            for y in other.points:
                self.boxes.append([x[0] + y, x[1]])

    def cartesian_mul(self, other):
        self_boxes = self.boxes
        self.boxes = []
        if not isinstance(other, Points):
            for x in self_boxes:
                for y in other.boxes:
                    lower_x, upper_x = x[0] - x[1], x[0] + x[1]
                    lower_y, upper_y = y[0] - y[1], y[0] + y[1]
                    tmp = [lower_x * lower_y, lower_x * upper_y, upper_x * lower_y, upper_x * upper_y]
                    lower = np.min(tmp, axis=0)
                    upper = np.max(tmp, axis=0)
                    self.boxes.append([(lower + upper) / 2, (upper - lower) / 2])

        for x in self_boxes:
            for y in other.points:
                self.boxes.append([x[0] * y, x[1] * np.abs(y)])

    def GRU_merge(self, self_act, aa, bb, act):
        self_boxes = self.boxes
        self.boxes = []
        for x in self_boxes:
            for a in aa.points:
                for b in bb.points:
                    self.boxes.append(GRU_merge(x, self_act, a, b, act))
                for b in bb.boxes:
                    self.boxes.append(GRU_merge(x, self_act, a, b, act))

            for a in aa.boxes:
                for b in bb.points:
                    self.boxes.append(GRU_merge(x, self_act, a, b, act))
                for b in bb.boxes:
                    self.boxes.append(GRU_merge(x, self_act, a, b, act))

    def __len__(self):
        return len(self.boxes)

    def matmul(self, W):
        for i in range(len(self.boxes)):
            self.boxes[i][0] = np.matmul(self.boxes[i][0], W)
            self.boxes[i][1] = np.matmul(self.boxes[i][1], np.abs(W))

    def bias_add(self, b):
        for i in range(len(self.boxes)):
            self.boxes[i][0] += b

    def activation(self, act):
        for i in range(len(self.boxes)):
            lower = act(self.boxes[i][0] - self.boxes[i][1])
            upper = act(self.boxes[i][0] + self.boxes[i][1])
            self.boxes[i] = [(lower + upper) / 2, (upper - lower) / 2]

    def __getitem__(self, item):
        return [[x[0][item], x[1][item]] for x in self.boxes]

    def __iter__(self):
        return self.boxes.__iter__()
