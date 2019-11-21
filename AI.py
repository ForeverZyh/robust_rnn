import keras.backend as K
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np


class AI:
    def __init__(self, center, radius, errors, same_shape):
        self.c = center
        self.r = radius
        if not same_shape:
            self.e = errors if errors is not None and errors.shape[-1] != 0 else None
            if self.e is not None:
                num_e = self.e.shape[-1] // self.c.shape[-1]
                mid = []
                for x in self.c.shape[1:]:
                    mid.append(int(x))
                self.e = K.reshape(self.e, [-1] + mid + [num_e])  # batch, c[1], ..., c[-1], num_e
                input_perm = [len(self.e.shape) - 1] + list(np.arange(len(self.e.shape) - 1))
                self.e = tf.transpose(self.e, perm=input_perm)  # num_e, batch, c[1], ..., c[-1]
        else:
            self.e = errors

        # print("init", self.c.shape, self.r.shape, None if self.e is None else self.e.shape)

    def matmul(self, W):
        # with tf.name_scope("matmal"):
        self.c = K.dot(self.c, W)
        self.r = K.dot(self.r, K.abs(W))
        self.e = None if self.e is None else K.dot(self.e, W)
        return self

    def bias_add(self, b):
        self.c = K.bias_add(self.c, b)
        return self

    def __add__(self, other):
        # print("add")
        if isinstance(other, AI):
            if self.e is None:
                return AI(self.c + other.c, self.r + other.r, other.e, True)
            elif other.e is None:
                return AI(self.c + other.c, self.r + other.r, self.e, True)
            else:
                return AI(self.c + other.c, self.r + other.r, K.concatenate([self.e, other.e], axis=0), True)
        else:
            return AI(self.c + other, self.r, self.e, True)

    def __mul__(self, other):
        # print("mul")
        if isinstance(other, AI):
            # raise NotImplementedError()
            lower_1, upper_1 = self.get_lu()
            lower_2, upper_2 = other.get_lu()
            all = K.stack([lower_1 * lower_2, lower_1 * upper_2, upper_1 * lower_2, upper_1 * upper_2], axis=0)
            lower = K.min(all, axis=0)
            upper = K.max(all, axis=0)
            return AI((lower + upper) / 2, (upper - lower) / 2, None, True)
        else:
            return AI(self.c * other, self.r * abs(other), None if self.e is None else self.e * other, True)

    def __truediv__(self, k):
        k = 1.0 / k
        return self * k

    def __sub__(self, other):
        return self + other * (-1)

    def get_lu(self):
        if self.e is None:
            return self.c - self.r, self.c + self.r
        else:
            return self.c - self.r - K.sum(K.abs(self.e), axis=0), self.c + self.r + K.sum(K.abs(self.e), axis=0)

    def activation(self, act):
        lower, upper = self.get_lu()
        lower = act(lower)
        upper = act(upper)
        self.c = (lower + upper) / 2
        self.r = (upper - lower) / 2
        self.e = None
        return self

    # def squeeze(self, axis):
    #     self.c = K.squeeze(self.c, axis=axis)
    #     self.r = K.squeeze(self.r, axis=axis)
    #     if self.e is not None:
    #         self.e = K.squeeze(self.e, axis=axis if axis < 0 else axis + 1)
    #     return self

    def portion_sum(self, other):
        lower_1, upper_1 = self.get_lu()
        lower_2, upper_2 = other.get_lu()
        all = K.stack([lower_1 * lower_2, lower_1 * upper_2, upper_1 * lower_2, upper_1 * upper_2], axis=0)
        lower = K.sum(K.min(all, axis=0), axis=1)
        upper = K.sum(K.max(all, axis=0), axis=1)
        self.c = (lower + upper) / 2
        self.r = (upper - lower) / 2
        self.e = None
        return self

    def softmax(self, axis):
        lower, upper = self.get_lu()
        max_upper = K.max(upper, axis=axis, keepdims=True)
        max_lower = K.max(lower, axis=axis, keepdims=True)
        new_lower = K.exp(lower - max_upper) / K.sum(K.exp(upper - max_upper), axis=axis, keepdims=True)
        new_upper = K.minimum(1.0, K.exp(upper - max_lower) / K.sum(K.exp(lower - max_lower), axis=axis, keepdims=True))
        self.c = (new_lower + new_upper) / 2
        self.r = (new_upper - new_lower) / 2
        self.e = None
        return self

    def to_state(self):
        if self.e is None:
            # print("to_state", self.c.shape, self.r.shape)
            return K.concatenate([self.c, self.r], axis=-1)
        else:
            # num_e, batch, c[1], ..., c[-1]
            input_perm = list(np.arange(1, len(self.e.shape))) + [0]
            errors = tf.transpose(self.e, perm=input_perm)  # batch, c[1], ..., c[-1], num_e
            num_e = errors.shape[-1]
            mid = []
            for x in self.c.shape[1:-1]:
                mid.append(int(x))
            # print("to_state", self.c.shape, self.e.shape)
            return K.concatenate(
                [self.c, self.r, K.reshape(errors, [-1] + mid + [num_e * self.c.shape[-1]])],
                axis=-1)  # batch, c[1], ..., c[-1]*num_e

    def GRU_merge(self, self_act, a, b, act):
        lower_a, upper_a = a.get_lu()
        lower_b, upper_b = b.get_lu()
        fa_lower, fa_upper = lower_a, upper_a
        fb_lower, fb_upper = act(lower_b), act(upper_b)
        lower_x, upper_x = self.get_lu()
        fx_lower, fx_upper = self_act(lower_x), self_act(upper_x)
        partial_fx_lower = tf.gradients(fx_lower, lower_x)[0]
        partial_fx_upper = tf.gradients(fx_upper, upper_x)[0]

        def lower_a_greater_zero():
            uz_x_Phi = K.minimum(partial_fx_upper * fa_upper, (fx_upper - fx_lower) * fa_upper / (upper_x - lower_x))
            ax_right_upper = fx_upper * fa_upper
            ax_left_upper = uz_x_Phi * (lower_x - upper_x) + ax_right_upper
            lz_x_Phi = K.minimum(partial_fx_lower * fa_lower, (fx_lower - fx_upper) * fa_lower / (lower_x - upper_x))
            ax_left_lower = fx_lower * fa_lower
            ax_right_lower = lz_x_Phi * (upper_x - lower_x) + ax_left_lower
            return [ax_left_lower, ax_left_upper, ax_right_lower, ax_right_upper]

        def lower_b_greater_zero():
            uz_x_Phi = K.maximum(-partial_fx_lower * fb_upper, (-fx_upper + fx_lower) * fb_upper / (upper_x - lower_x))
            bx_left_upper = (1 - fx_lower) * fb_upper
            bx_right_upper = uz_x_Phi * (upper_x - lower_x) + bx_left_upper
            lz_x_Phi = K.maximum(-partial_fx_upper * fb_lower, (-fx_lower + fx_upper) * fb_lower / (lower_x - upper_x))
            bx_right_lower = (1 - fx_upper) * fb_lower
            bx_left_lower = lz_x_Phi * (lower_x - upper_x) + bx_right_lower
            return [bx_left_lower, bx_left_upper, bx_right_lower, bx_right_upper]

        def upper_a_less_zero():
            uz_x_Phi = K.maximum(partial_fx_lower * fa_upper, (fx_lower - fx_upper) * fa_upper / (lower_x - upper_x))
            ax_left_upper = fx_lower * fa_upper
            ax_right_upper = uz_x_Phi * (upper_x - lower_x) + ax_left_upper
            lz_x_Phi = K.maximum(partial_fx_upper * fa_lower, (fx_upper - fx_lower) * fa_lower / (upper_x - lower_x))
            ax_right_lower = fx_upper * fa_lower
            ax_left_lower = lz_x_Phi * (lower_x - upper_x) + ax_right_lower
            return [ax_left_lower, ax_left_upper, ax_right_lower, ax_right_upper]

        def upper_b_less_zero():
            uz_x_Phi = K.minimum(-partial_fx_upper * fb_upper, (-fx_upper + fx_lower) * fb_upper / (upper_x - lower_x))
            bx_right_upper = (1 - fx_upper) * fb_upper
            bx_left_upper = uz_x_Phi * (lower_x - upper_x) + bx_right_upper
            lz_x_Phi = K.minimum(-partial_fx_lower * fb_lower, (-fx_lower + fx_upper) * fb_lower / (lower_x - upper_x))
            bx_left_lower = (1 - fx_lower) * fb_lower
            bx_right_lower = lz_x_Phi * (upper_x - lower_x) + bx_left_lower
            return [bx_left_lower, bx_left_upper, bx_right_lower, bx_right_upper]

        def otherwise_a():
            uz_x_Phi = K.minimum(partial_fx_upper * fa_upper, (fx_upper - fx_lower) * fa_upper / (upper_x - lower_x))
            ax_right_upper = fx_upper * fa_upper
            ax_left_upper = uz_x_Phi * (lower_x - upper_x) + ax_right_upper
            lz_x_Phi = K.maximum(partial_fx_upper * fa_lower, (fx_upper - fx_lower) * fa_lower / (upper_x - lower_x))
            ax_right_lower = fx_upper * fa_lower
            ax_left_lower = lz_x_Phi * (lower_x - upper_x) + ax_right_lower
            return [ax_left_lower, ax_left_upper, ax_right_lower, ax_right_upper]

        def otherwise_b():
            uz_x_Phi = K.maximum(-partial_fx_lower * fb_upper, (-fx_upper + fx_lower) * fb_upper / (upper_x - lower_x))
            bx_left_upper = (1 - fx_lower) * fb_upper
            bx_right_upper = uz_x_Phi * (upper_x - lower_x) + bx_left_upper
            lz_x_Phi = K.minimum(-partial_fx_lower * fb_lower, (-fx_lower + fx_upper) * fb_lower / (lower_x - upper_x))
            bx_left_lower = (1 - fx_lower) * fb_lower
            bx_right_lower = lz_x_Phi * (upper_x - lower_x) + bx_left_lower
            return [bx_left_lower, bx_left_upper, bx_right_lower, bx_right_upper]

        a_anchors = otherwise_a()
        anchors_lower_a_greater_zero = lower_a_greater_zero()
        anchors_upper_a_less_zero = upper_a_less_zero()
        for i in range(4):
            a_anchors[i] = K.switch(K.greater(lower_a, K.zeros_like(lower_a)), anchors_lower_a_greater_zero[i],
                                    a_anchors[i])
            a_anchors[i] = K.switch(K.less(upper_a, K.zeros_like(upper_a)), anchors_upper_a_less_zero[i], a_anchors[i])

        b_anchors = otherwise_b()
        anchors_lower_b_greater_zero = lower_b_greater_zero()
        anchors_upper_b_less_zero = upper_b_less_zero()
        for i in range(4):
            b_anchors[i] = K.switch(K.greater(lower_b, K.zeros_like(lower_b)), anchors_lower_b_greater_zero[i],
                                    b_anchors[i])
            b_anchors[i] = K.switch(K.less(upper_b, K.zeros_like(upper_b)), anchors_upper_b_less_zero[i], b_anchors[i])

        for i in range(4):
            a_anchors[i] += b_anchors[i]
        lower_z = K.minimum(a_anchors[0], a_anchors[2])
        upper_z = K.maximum(a_anchors[1], a_anchors[3])
        return AI((lower_z + upper_z) / 2, (upper_z - lower_z) / 2, None, True)

    def GRU_merge1(self, self_act, a, b, act):
        lower_a, upper_a = a.get_lu()
        lower_b, upper_b = b.get_lu()
        fa_lower, fa_upper = lower_a, upper_a
        fb_lower, fb_upper = act(lower_b), act(upper_b)
        lower_x, upper_x = self.get_lu()
        fx_lower, fx_upper = self_act(lower_x), self_act(upper_x)

        def lower_a_greater_zero():
            return [fx_lower * fa_lower, fx_lower * fa_upper, fx_upper * fa_lower, fx_upper * fa_upper]

        def lower_b_greater_zero():
            return [(1 - fx_lower) * fb_lower, (1 - fx_lower) * fb_upper, (1 - fx_upper) * fb_lower, (
                    1 - fx_upper) * fb_upper]

        abounds = [fa_lower, fa_upper]
        bbounds = [fb_lower, fb_upper]
        xbounds = [fx_lower, fx_upper]
        rets = []
        for ab in abounds:
            for bb in bbounds:
                for xb in xbounds:
                    rets.append(K.expand_dims(ab * xb + (1 - xb) * bb, axis=-1))

        rets = K.concatenate(rets, axis=-1)
        lower_z = K.min(rets, axis=-1)
        upper_z = K.max(rets, axis=-1)
        return AI((lower_z + upper_z) / 2, (upper_z - lower_z) / 2, None, True)

    # def mul(self, other, f1, f2, f1_dec=False):
    #     # f1, f2 are two monotonic functions, and f1(x) > 0
    #     # f1_dec is True means that f1 decreases with respect to x, otherwise increases
    #     # f2 always increases with respect to y
    #
    #     lower_1, upper_1 = self.get_lu()
    #     if f1_dec:
    #         f1_lower = f1(lower_1)
    #         partial_f1_lower = tf.gradients(f1_lower, lower_1)
    #         f1_upper = f1(upper_1)
    #         partial_f1_upper = tf.gradients(f1_upper, upper_1)
    #     else:
    #         f1_lower = f1(upper_1)
    #         partial_f1_lower = tf.gradients(f1_lower, upper_1)
    #         f1_upper = f1(lower_1)
    #         partial_f1_upper = tf.gradients(f1_upper, lower_1)
    #     lower_2, upper_2 = other.get_lu()
    #     f2_lower = f2(lower_2)
    #     partial_f2_lower = tf.gradients(f2_lower, lower_2)
    #     f2_upper = f2(upper_2)
    #     partial_f2_upper = tf.gradients(f2_upper, upper_2)
    #
    #     def lower_2_greater_zero():
    #         if not f1_dec:
    #             uz_x_Phi = K.minimum(partial_f1_upper * f2_upper,
    #                                  (f1_upper - f1_lower) * f2_upper / (upper_1 - lower_1))
    #             uz_x_ax = upper_1
    #             uz_x_b = f1_upper * f2_upper
    #             lz_x_Phi = K.minimum(partial_f1_lower * f2_lower,
    #                                  (f1_lower - f1_upper) * f2_lower / (lower_1 - upper_1))
    #             lz_x_ax = lower_1
    #             lz_x_b = f1_lower * f2_lower
    #             z_x_Phi = K.minimum(uz_x_Phi, lz_x_Phi)
    #             volx = (uz_x_b - lz_x_b) * (upper_1 - lower_1) - (upper_1 - lower_1) * (upper_1 - lower_1) * z_x_Phi * 2
    #         else:
    #             uz_x_Phi = K.maximum(partial_f1_upper * f2_upper,
    #                                  (f1_upper - f1_lower) * f2_upper / (lower_1 - upper_1))
    #             uz_x_ax = lower_1
    #             uz_x_b = f1_upper * f2_upper
    #             lz_x_Phi = K.maximum(partial_f1_lower * f2_lower,
    #                                  (f1_lower - f1_upper) * f2_lower / (upper_1 - lower_1))
    #             lz_x_ax = upper_1
    #             lz_x_b = f1_lower * f2_lower
    #             z_x_Phi = K.maximum(uz_x_Phi, lz_x_Phi)
    #             volx = (uz_x_b - lz_x_b) * (upper_1 - lower_1) + (upper_1 - lower_1) * (upper_1 - lower_1) * z_x_Phi * 2
    #
    #         uz_y_Phi = K.minimum(partial_f2_upper * f1_upper, (f2_upper - f2_lower) * f1_upper / (upper_2 - lower_2))
    #         uz_y_ay = upper_2
    #         uz_y_b = f1_upper * f2_upper
    #         lz_y_Phi = K.minimum(partial_f2_lower * f1_lower, (f2_lower - f2_upper) * f1_lower / (lower_2 - upper_2))
    #         lz_y_ay = lower_2
    #         lz_y_b = f1_lower * f2_lower
    #         z_y_Phi = K.minimum(uz_y_Phi, lz_y_Phi)
    #         voly = (uz_y_b - lz_y_b) * (upper_2 - lower_2) - (upper_2 - lower_2) * (upper_2 - lower_2) * z_y_Phi * 2
    #         ret_x = K.concatenate([z_x_Phi, uz_x_b, ])
    #
    #     def upper_2_less_zero():
    #         pass
    #
    #     def otherwise():
    #         pass

    def __getitem__(self, item):
        errors = None
        if self.e is not None:
            # print(self.e.shape)
            errors = [K.expand_dims(self.e[i][item], axis=0) for i in range(int(self.e.shape[0]))]
            errors = K.concatenate(errors, axis=0)

        # print("getitem")
        return AI(self.c[item], self.r[item], errors, True)
