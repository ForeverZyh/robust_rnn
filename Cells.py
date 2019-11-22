import keras.backend as K
from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
import keras
from keras.engine.base_layer import Layer
from keras.engine.base_layer import disable_tracking
from keras.engine.base_layer import InputSpec
from keras.utils.generic_utils import has_arg
from keras.utils.generic_utils import to_list
import numpy as np
from rnn_verification.AI import AI
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class AIGRUCell(Layer):
    """Cell class for the GRU layer.
    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
            Default: hard sigmoid (`hard_sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        implementation: Implementation mode, either 1 or 2.
            Mode 1 will structure its operations as a larger number of
            smaller dot products and additions, whereas mode 2 will
            batch them into fewer, larger operations. These modes will
            have different performance profiles on different hardware and
            for different applications.
        reset_after: GRU convention (whether to apply reset gate after or
            before matrix multiplication). False = "before" (default),
            True = "after" (CuDNN compatible).
    """

    def __init__(self, units, input_dim,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer=None,
                 recurrent_initializer=None,
                 bias_initializer=None,
                 reset_after=False,
                 **kwargs):
        super(AIGRUCell, self).__init__(**kwargs)
        self.units = units
        self.input_dim = input_dim
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer

        self.kernel_regularizer = regularizers.get(None)
        self.recurrent_regularizer = regularizers.get(None)
        self.bias_regularizer = regularizers.get(None)

        self.kernel_constraint = constraints.get(None)
        self.recurrent_constraint = constraints.get(None)
        self.bias_constraint = constraints.get(None)

        self.dropout = .0
        self.recurrent_dropout = .0
        self.implementation = 2
        self.reset_after = reset_after
        self.state_size = self.units * 2
        self.output_size = self.units * 2
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def build(self, input_shape):

        if isinstance(self.recurrent_initializer, initializers.Identity):
            def recurrent_identity(shape, gain=1., dtype=None):
                del dtype
                return gain * np.concatenate(
                    [np.identity(shape[0])] * (shape[1] // shape[0]), axis=1)

            self.recurrent_initializer = recurrent_identity

        self.kernel = self.kernel_initializer
        self.recurrent_kernel = self.recurrent_initializer

        if self.use_bias:
            if not self.reset_after:
                bias_shape = (3 * self.units,)
            else:
                # separate biases for input and recurrent kernels
                # Note: the shape is intentionally different from CuDNNGRU biases
                # `(2 * 3 * self.units,)`, so that we can distinguish the classes
                # when loading and converting saved weights.
                bias_shape = (2, 3 * self.units)
            self.bias = self.bias_initializer
            if not self.reset_after:
                self.input_bias, self.recurrent_bias = self.bias, None
            else:
                # NOTE: need to flatten, since slicing in CNTK gives 2D array
                self.input_bias = K.flatten(self.bias[0])
                self.recurrent_bias = K.flatten(self.bias[1])
        else:
            self.bias = None

        # update gate
        self.kernel_z = self.kernel[:, :self.units]
        self.recurrent_kernel_z = self.recurrent_kernel[:, :self.units]
        # reset gate
        self.kernel_r = self.kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_r = self.recurrent_kernel[:,
                                  self.units:
                                  self.units * 2]
        # new gate
        self.kernel_h = self.kernel[:, self.units * 2:]
        self.recurrent_kernel_h = self.recurrent_kernel[:, self.units * 2:]

        if self.use_bias:
            # bias for inputs
            self.input_bias_z = self.input_bias[:self.units]
            self.input_bias_r = self.input_bias[self.units: self.units * 2]
            self.input_bias_h = self.input_bias[self.units * 2:]
            # bias for hidden state - just for compatibility with CuDNN
            if self.reset_after:
                self.recurrent_bias_z = self.recurrent_bias[:self.units]
                self.recurrent_bias_r = (
                    self.recurrent_bias[self.units: self.units * 2])
                self.recurrent_bias_h = self.recurrent_bias[self.units * 2:]
        else:
            self.input_bias_z = None
            self.input_bias_r = None
            self.input_bias_h = None
            if self.reset_after:
                self.recurrent_bias_z = None
                self.recurrent_bias_r = None
                self.recurrent_bias_h = None
        self.built = True

    def call(self, _inputs, _states, training=None):
        # print(_inputs.shape, _states[0].shape)
        h_tm1_c = _states[0][:, :self.units]  # the center of the last state
        h_tm1_r = _states[0][:, self.units:self.units * 2]  # the radius of the last state
        h_tm1_e = _states[0][:, self.units * 2:]  # the errors of the last state
        inputs_c = _inputs[:, :self.input_dim]  # the center of the inputs
        inputs_r = _inputs[:, self.input_dim:self.input_dim * 2]  # the radius of the inputs
        inputs_e = _inputs[:, self.input_dim * 2:]  # the errors of the last state
        h_tm1 = AI(h_tm1_c, h_tm1_r, h_tm1_e, False)
        inputs = AI(inputs_c, inputs_r, inputs_e, False)
        matrix_inner = AI(h_tm1_c, h_tm1_r, h_tm1_e, False)
        # print(inputs_c, inputs_r, inputs_e)

        if self.implementation == 1:
            raise NotImplementedError()
        else:
            # inputs projected by all gate matrices at once
            inputs.matmul(self.kernel)
            if self.use_bias:
                # biases: bias_z_i, bias_r_i, bias_h_i
                inputs.bias_add(self.input_bias)
            x_z = inputs[:, :self.units]
            x_r = inputs[:, self.units: 2 * self.units]
            x_h = inputs[:, 2 * self.units:]

            if self.reset_after:
                # hidden state projected by all gate matrices at once
                matrix_inner.matmul(self.recurrent_kernel)
                if self.use_bias:
                    matrix_inner.bias_add(self.recurrent_bias)
            else:
                # hidden state projected separately for update/reset and new
                matrix_inner.matmul(self.recurrent_kernel[:, :2 * self.units])

            recurrent_z = matrix_inner[:, :self.units]
            recurrent_r = matrix_inner[:, self.units: 2 * self.units]

            z = x_z + recurrent_z
            # z.activation(self.recurrent_activation)
            r = x_r + recurrent_r
            r.activation(self.recurrent_activation)

            if self.reset_after:
                recurrent_h = r * matrix_inner[:, 2 * self.units:]
            else:
                recurrent_h = r * h_tm1
                recurrent_h.matmul(self.recurrent_kernel[:, 2 * self.units:])

            hh = x_h + recurrent_h
            # hh.activation(self.activation)

        # previous and candidate state mixed by update gate
        h = z.GRU_merge1(self.recurrent_activation, h_tm1, hh, self.activation)

        h_state = h.to_state()
        return h_state, [h_state]

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation':
                      activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer':
                      self.kernel_initializer,
                  'recurrent_initializer':
                      self.recurrent_initializer,
                  'bias_initializer': self.bias_initializer,
                  'kernel_regularizer':
                      regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer':
                      regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint':
                      constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'implementation': self.implementation,
                  'reset_after': self.reset_after}
        base_config = super(AIGRUCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DPAIGRUCell(Layer):
    """Cell class for the GRU layer.
    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
            Default: hard sigmoid (`hard_sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        implementation: Implementation mode, either 1 or 2.
            Mode 1 will structure its operations as a larger number of
            smaller dot products and additions, whereas mode 2 will
            batch them into fewer, larger operations. These modes will
            have different performance profiles on different hardware and
            for different applications.
        reset_after: GRU convention (whether to apply reset gate after or
            before matrix multiplication). False = "before" (default),
            True = "after" (CuDNN compatible).
    """

    def __init__(self, units, input_dim,
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer=None,
                 recurrent_initializer=None,
                 bias_initializer=None,
                 reset_after=False,
                 **kwargs):
        super(DPAIGRUCell, self).__init__(**kwargs)
        self.units = units
        self.input_dim = input_dim
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer

        self.kernel_regularizer = regularizers.get(None)
        self.recurrent_regularizer = regularizers.get(None)
        self.bias_regularizer = regularizers.get(None)

        self.kernel_constraint = constraints.get(None)
        self.recurrent_constraint = constraints.get(None)
        self.bias_constraint = constraints.get(None)

        self.dropout = .0
        self.recurrent_dropout = .0
        self.implementation = 2
        self.reset_after = reset_after
        self.state_size = self.units * 2
        self.output_size = self.units * 2
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def build(self, input_shape):

        if isinstance(self.recurrent_initializer, initializers.Identity):
            def recurrent_identity(shape, gain=1., dtype=None):
                del dtype
                return gain * np.concatenate(
                    [np.identity(shape[0])] * (shape[1] // shape[0]), axis=1)

            self.recurrent_initializer = recurrent_identity

        self.kernel = self.kernel_initializer
        self.recurrent_kernel = self.recurrent_initializer

        if self.use_bias:
            if not self.reset_after:
                bias_shape = (3 * self.units,)
            else:
                # separate biases for input and recurrent kernels
                # Note: the shape is intentionally different from CuDNNGRU biases
                # `(2 * 3 * self.units,)`, so that we can distinguish the classes
                # when loading and converting saved weights.
                bias_shape = (2, 3 * self.units)
            self.bias = self.bias_initializer
            if not self.reset_after:
                self.input_bias, self.recurrent_bias = self.bias, None
            else:
                # NOTE: need to flatten, since slicing in CNTK gives 2D array
                self.input_bias = K.flatten(self.bias[0])
                self.recurrent_bias = K.flatten(self.bias[1])
        else:
            self.bias = None

        # update gate
        self.kernel_z = self.kernel[:, :self.units]
        self.recurrent_kernel_z = self.recurrent_kernel[:, :self.units]
        # reset gate
        self.kernel_r = self.kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_r = self.recurrent_kernel[:,
                                  self.units:
                                  self.units * 2]
        # new gate
        self.kernel_h = self.kernel[:, self.units * 2:]
        self.recurrent_kernel_h = self.recurrent_kernel[:, self.units * 2:]

        if self.use_bias:
            # bias for inputs
            self.input_bias_z = self.input_bias[:self.units]
            self.input_bias_r = self.input_bias[self.units: self.units * 2]
            self.input_bias_h = self.input_bias[self.units * 2:]
            # bias for hidden state - just for compatibility with CuDNN
            if self.reset_after:
                self.recurrent_bias_z = self.recurrent_bias[:self.units]
                self.recurrent_bias_r = (
                    self.recurrent_bias[self.units: self.units * 2])
                self.recurrent_bias_h = self.recurrent_bias[self.units * 2:]
        else:
            self.input_bias_z = None
            self.input_bias_r = None
            self.input_bias_h = None
            if self.reset_after:
                self.recurrent_bias_z = None
                self.recurrent_bias_r = None
                self.recurrent_bias_h = None
        self.built = True

    def call(self, _inputs, _states, training=None):
        # print(_inputs.shape, _states[0].shape)
        h_tm1_c = _states[0][:, :self.units]  # the center of the last state
        h_tm1_r = _states[0][:, self.units:self.units * 2]  # the radius of the last state
        h_tm1_e = _states[0][:, self.units * 2:]  # the errors of the last state
        h_tm1_without_modify_c = _inputs[:, 0:self.units]
        h_tm1_without_modify_r = _inputs[:, self.units:self.units * 2]
        h_tm1_without_modify_e = None  # We have already assumed that this is a Box Domain
        h_without_dp_c = _inputs[:, self.units * 2:self.units * 3]
        h_without_dp_r = _inputs[:, self.units * 3:self.units * 4]
        h_without_dp_e = None  # We have already assumed that this is a Box Domain
        inputs_without_modify_c = _inputs[:, self.units * 4:self.units * 4 + self.input_dim]
        inputs_without_modify_r = _inputs[:, self.units * 4 + self.input_dim:self.units * 4 + self.input_dim * 2]
        inputs_without_modify_e = None  # This is a Point Domain
        inputs_c = _inputs[:,
                   self.units * 4 + self.input_dim * 2:self.units * 4 + self.input_dim * 3]  # the center of the inputs
        inputs_r = _inputs[:,
                   self.units * 4 + self.input_dim * 3:self.units * 4 + self.input_dim * 4]  # the radius of the inputs
        inputs_e = _inputs[:, self.units * 4 + self.input_dim * 4:]  # the errors of the last state

        def cell(h_tm1_c, h_tm1_r, h_tm1_e, inputs_c, inputs_r, inputs_e):
            h_tm1 = AI(h_tm1_c, h_tm1_r, h_tm1_e, False)
            inputs = AI(inputs_c, inputs_r, inputs_e, False)
            matrix_inner = AI(h_tm1_c, h_tm1_r, h_tm1_e, False)
            # print(inputs_c, inputs_r, inputs_e)

            if self.implementation == 1:
                raise NotImplementedError()
            else:
                # inputs projected by all gate matrices at once
                inputs.matmul(self.kernel)
                if self.use_bias:
                    # biases: bias_z_i, bias_r_i, bias_h_i
                    inputs.bias_add(self.input_bias)
                x_z = inputs[:, :self.units]
                x_r = inputs[:, self.units: 2 * self.units]
                x_h = inputs[:, 2 * self.units:]

                if self.reset_after:
                    # hidden state projected by all gate matrices at once
                    matrix_inner.matmul(self.recurrent_kernel)
                    if self.use_bias:
                        matrix_inner.bias_add(self.recurrent_bias)
                else:
                    # hidden state projected separately for update/reset and new
                    matrix_inner.matmul(self.recurrent_kernel[:, :2 * self.units])

                recurrent_z = matrix_inner[:, :self.units]
                recurrent_r = matrix_inner[:, self.units: 2 * self.units]

                z = x_z + recurrent_z
                # z.activation(self.recurrent_activation)
                r = x_r + recurrent_r
                r.activation(self.recurrent_activation)

                if self.reset_after:
                    recurrent_h = r * matrix_inner[:, 2 * self.units:]
                else:
                    recurrent_h = r * h_tm1
                    recurrent_h.matmul(self.recurrent_kernel[:, 2 * self.units:])

                hh = x_h + recurrent_h
                # hh.activation(self.activation)

            # previous and candidate state mixed by update gate
            h = z.GRU_merge1(self.recurrent_activation, h_tm1, hh, self.activation)
            return h

        h1_lower, h1_upper = cell(h_tm1_c, h_tm1_r, h_tm1_e, inputs_without_modify_c, inputs_without_modify_r,
                                  inputs_without_modify_e).get_lu()
        h2_lower, h2_upper = cell(h_tm1_without_modify_c, h_tm1_without_modify_r, h_tm1_without_modify_e, inputs_c,
                                  inputs_r, inputs_e).get_lu()
        h_without_dp_lower = h_without_dp_c - h_without_dp_r
        h_without_dp_upper = h_without_dp_c + h_without_dp_r
        lower = K.maximum(K.minimum(h1_lower, h2_lower), h_without_dp_lower)
        upper = K.minimum(K.maximum(h1_upper, h2_upper), h_without_dp_upper)
        # lower = K.minimum(h1_lower, h2_lower)
        # upper = K.maximum(h1_upper, h2_upper)
        h_state = K.concatenate([(lower + upper) / 2, (upper - lower) / 2], axis=-1)
        return h_state, [h_state]

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation':
                      activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer':
                      self.kernel_initializer,
                  'recurrent_initializer':
                      self.recurrent_initializer,
                  'bias_initializer': self.bias_initializer,
                  'kernel_regularizer':
                      regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer':
                      regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint':
                      constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'implementation': self.implementation,
                  'reset_after': self.reset_after}
        base_config = super(AIGRUCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Zonotope(keras.layers.Layer):
    def __init__(self, delta, fi, **kwargs):
        super(Zonotope, self).__init__(**kwargs)
        self.delta = delta
        self.fi = fi

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(Zonotope, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, **kwargs):
        # x has (batch, length, dim)
        delta = self.delta
        xshape = x.shape
        r = K.zeros_like(x[:, 0, ])
        rets = []
        for i in range(xshape[1]):
            if self.fi == -1 or i < self.fi:
                errors = []
                mid = x[:, i, :]
                cnt = 0
                for j in range(-delta, delta + 1):
                    if 0 <= i + j < xshape[1] and j != 0:
                        mid = (mid + x[:, i + j, :]) / 2
                        errors.append((mid - x[:, i + j, :]) / 2)
                        cnt += 1
                for _ in range(delta * 2 - cnt):
                    errors.append(r)
                rets.append(K.expand_dims(K.concatenate([x[:, i, :], r] + errors, axis=-1), axis=1))
            else:
                rets.append(
                    K.expand_dims(K.concatenate([x[:, i, :]] + [r for _ in range(delta * 2 + 1)], axis=-1), axis=1))
        return K.concatenate(rets, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2] * (2 * self.delta + 2))


class Points(keras.layers.Layer):
    def __init__(self, delta, fi, **kwargs):
        super(Points, self).__init__(**kwargs)
        self.delta = delta
        self.fi = fi

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(Points, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, **kwargs):
        # x has (batch, length, dim)
        delta = self.delta
        xshape = x.shape
        r = K.zeros_like(x[:, 0, ])
        rets = []
        for i in range(xshape[1]):
            if self.fi == -1 or i < self.fi:
                errors = []
                cnt = 0
                for j in range(-delta, delta + 1):
                    if 0 <= i + j < xshape[1] and j != 0:
                        errors.append(x[:, i + j, :])
                        cnt += 1
                for _ in range(delta * 2 + 1 - cnt):
                    errors.append(x[:, i, :])
                rets.append(K.expand_dims(K.concatenate(errors, axis=-1), axis=1))
            else:
                rets.append(
                    K.expand_dims(K.concatenate([x[:, i, :] for _ in range(delta * 2 + 1)], axis=-1), axis=1))
        return K.concatenate(rets, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2] * (2 * self.delta + 1))


def _generate_dropout_mask(ones, rate, training=None, count=1):
    def dropped_inputs():
        return K.dropout(ones, rate)

    if count > 1:
        return [K.in_train_phase(
            dropped_inputs,
            ones,
            training=training) for _ in range(count)]
    return K.in_train_phase(
        dropped_inputs,
        ones,
        training=training)
