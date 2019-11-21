import keras.backend as K
from keras.models import Model
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from keras.layers import Embedding, GRU, RNN, Input, Dense, Lambda, MaxPooling1D
from keras.backend import squeeze
import keras
import numpy as np

from rnn_verification.Cells import AIGRUCell, Zonotope, DPAIGRUCell
from rnn_verification.AI import AI


class SimpleRNNModel1:
    def __init__(self, hyperparameters, all_voc_size, nb_classes):
        self.D = hyperparameters["D"]

        self.hyperparameters = hyperparameters
        self.__check_all_hyperparmeters_exist()
        self.all_voc_size = all_voc_size
        self.nb_classes = nb_classes

        # self.__init_parameter(empirical_name_dist)
        self.build()

    @staticmethod
    def uniform_initializer(scale):
        return keras.initializers.RandomUniform(minval=0, maxval=scale)

    def build(self):
        self.c = Input(shape=(self.hyperparameters["max_len"],), dtype='int32', name="input")
        embed_init = self.uniform_initializer(10 ** self.hyperparameters["log_name_rep_init_scale"])
        self.embed = Embedding(self.all_voc_size, self.D, embeddings_initializer=embed_init, name="embedding")
        look_up_c = self.embed(self.c)
        gru_init = self.uniform_initializer(10 ** self.hyperparameters["log_hidden_init_scale"])
        self.gru = GRU(self.hyperparameters["conv_layer2_nfilters"], return_sequences=False,
                       recurrent_initializer=gru_init,
                       kernel_initializer=gru_init, bias_initializer=gru_init, name="gru")
        self.h_t = self.gru(look_up_c)
        self.fc1 = Dense(self.nb_classes, activation='softmax')
        self.logits = self.fc1(self.h_t)
        # self.predictions = tf.argmax(self.logits, -1)
        #
        # self.loss = tf.reduce_mean(
        #     tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
        #
        # correct_predictions = tf.equal(self.predictions, tf.argmax(self.y, axis=-1))
        # self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        self.model = Model(inputs=self.c, outputs=self.logits)
        self.model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])
        self.early_stopping = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
                                                                      verbose=0, mode='auto',
                                                                      baseline=None, restore_best_weights=False)

    def __check_all_hyperparmeters_exist(self):
        all_params = ["D", "batch_size", "max_len",
                      "log_name_rep_init_scale", "log_hidden_init_scale", "conv_layer2_nfilters"]
        for param in all_params:
            assert param in self.hyperparameters, param

    def verify_swap(self, delta=1, fi=-1, ai="Point", dp=True):
        look_up_c = self.embed(self.c)
        kernel, recurrent_kernel, bias = self.gru.weights
        # kernel = tf.constant(kernel, name="const_kernel")
        # recurrent_kernel = tf.constant(recurrent_kernel)
        units = self.hyperparameters["conv_layer2_nfilters"]
        aigru = AIGRUCell(units, self.hyperparameters["D"], recurrent_initializer=recurrent_kernel,
                          kernel_initializer=kernel, bias_initializer=bias)
        domain = None
        if ai == "Box":
            maxpool = MaxPooling1D(pool_size=delta * 2 + 1, strides=1, padding='same')
            if fi != -1:
                box_domain = Lambda(
                    lambda x: K.concatenate(
                        [K.concatenate([(maxpool(x[:, :fi, :]) - maxpool(-x[:, :fi, :])) / 2, x[:, fi:, :]], axis=-2),
                         K.concatenate(
                             [(maxpool(x[:, :fi, :]) + maxpool(-x[:, :fi, :])) / 2, K.zeros_like(x[:, fi:, :])],
                             axis=-2)], axis=-1))(look_up_c)
            else:
                box_domain = Lambda(
                    lambda x: K.concatenate([(maxpool(x) - maxpool(-x)) / 2, (maxpool(x) + maxpool(-x)) / 2],
                                            axis=-1))(look_up_c)
            domain = box_domain
        elif ai == "Point":
            point_domain = Lambda(lambda x: K.concatenate([x, K.zeros_like(x)], axis=-1))(look_up_c)
            self.h_t_ai = RNN(aigru, return_sequences=False)(point_domain)
        elif ai == "Zonotope":
            zonotope_domain = Zonotope(delta, fi)(look_up_c)
            domain = zonotope_domain
        else:
            raise NotImplementedError()

        if ai != "Point":
            if dp:
                point_domain = Lambda(lambda x: K.concatenate([x, K.zeros_like(x)], axis=-1))(look_up_c)
                rnn_layer = RNN(aigru, return_sequences=True)
                h_t_ai_s = rnn_layer(point_domain)
                h_t_ai_without_dp = rnn_layer(domain)
                dpaigru = DPAIGRUCell(units, self.hyperparameters["D"], recurrent_initializer=recurrent_kernel,
                                      kernel_initializer=kernel, bias_initializer=bias)
                for i in range(delta * 2):
                    h_t_ai_s = Lambda(lambda x: K.concatenate(
                        [K.concatenate([K.zeros_like(x[:, :1, :]), x[:, 1:, :]], axis=1), h_t_ai_without_dp,
                         point_domain, domain], axis=-1))(h_t_ai_s)
                    h_t_ai_s = RNN(dpaigru, return_sequences=(i != 2 * delta - 1))(h_t_ai_s)
                self.h_t_ai = h_t_ai_s
            else:
                self.h_t_ai = RNN(aigru, return_sequences=False)(domain)

        weights, bias = self.fc1.weights
        to_AI = lambda d: (lambda x: AI(x[:, :d], x[:, d:d * 2], x[:, d * 2:], False))
        attention = Lambda(lambda x: to_AI(units)(x).matmul(weights).bias_add(bias).to_state())(self.h_t_ai)
        self.verify_model = Model(inputs=self.c, outputs=attention)


class SimpleRNNModel:
    def __init__(self, hyperparameters, all_voc_size, nb_classes):
        self.D = hyperparameters["D"]

        self.hyperparameters = hyperparameters
        self.__check_all_hyperparmeters_exist()
        self.all_voc_size = all_voc_size
        self.nb_classes = nb_classes

        # self.__init_parameter(empirical_name_dist)
        self.build()

    @staticmethod
    def uniform_initializer(scale):
        return keras.initializers.RandomUniform(minval=0, maxval=scale)

    def build(self):
        self.c = Input(shape=(self.hyperparameters["max_len"],), dtype='int32', name="input")
        embed_init = self.uniform_initializer(10 ** self.hyperparameters["log_name_rep_init_scale"])
        self.embed = Embedding(self.all_voc_size, self.D, embeddings_initializer=embed_init, name="embedding")
        look_up_c = self.embed(self.c)
        gru_init = self.uniform_initializer(10 ** self.hyperparameters["log_hidden_init_scale"])
        self.gru = GRU(self.hyperparameters["conv_layer2_nfilters"], return_sequences=True,
                       recurrent_initializer=gru_init,
                       kernel_initializer=gru_init, bias_initializer=gru_init, name="gru")
        self.h_t = self.gru(look_up_c)
        self.squeeze = Dense(1, name="attention_dense")
        attention_out = Lambda(lambda x: squeeze(
            tf.matmul(tf.transpose(x, perm=[0, 2, 1]), tf.nn.softmax(self.squeeze(x), axis=1)),
            axis=-1))(self.h_t)
        self.fc1 = Dense(self.nb_classes, activation='softmax')
        self.logits = self.fc1(attention_out)
        # self.predictions = tf.argmax(self.logits, -1)
        #
        # self.loss = tf.reduce_mean(
        #     tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
        #
        # correct_predictions = tf.equal(self.predictions, tf.argmax(self.y, axis=-1))
        # self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        self.model = Model(inputs=self.c, outputs=self.logits)
        self.model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])
        self.early_stopping = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5,
                                                                      verbose=0, mode='auto',
                                                                      baseline=None, restore_best_weights=False)

    def __check_all_hyperparmeters_exist(self):
        all_params = ["D", "batch_size", "max_len",
                      "log_name_rep_init_scale", "log_hidden_init_scale", "conv_layer2_nfilters"]
        for param in all_params:
            assert param in self.hyperparameters, param

    def verify_swap(self, delta=1, fi=-1, ai="Point"):
        look_up_c = self.embed(self.c)
        kernel, recurrent_kernel, bias = self.gru.weights
        units = self.hyperparameters["conv_layer2_nfilters"]
        aigru = AIGRUCell(units, self.hyperparameters["D"], recurrent_initializer=recurrent_kernel,
                          kernel_initializer=kernel, bias_initializer=bias)
        if ai == "Box":
            maxpool = MaxPooling1D(pool_size=delta * 2 + 1, strides=1, padding='same')
            if fi != -1:
                box_domain = Lambda(
                    lambda x: K.concatenate(
                        [K.concatenate([(maxpool(x[:, :fi, :]) - maxpool(-x[:, :fi, :])) / 2, x[:, fi:, :]], axis=-2),
                         K.concatenate(
                             [(maxpool(x[:, :fi, :]) + maxpool(-x[:, :fi, :])) / 2, K.zeros_like(x[:, fi:, :])],
                             axis=-2)], axis=-1))(look_up_c)
            else:
                box_domain = Lambda(
                    lambda x: K.concatenate([(maxpool(x) - maxpool(-x)) / 2, (maxpool(x) + maxpool(-x)) / 2],
                                            axis=-1))(look_up_c)
            self.h_t_ai = RNN(aigru, return_sequences=True)(box_domain)
        elif ai == "Point":
            point_domain = Lambda(lambda x: K.concatenate([x, K.zeros_like(x)], axis=-1))(look_up_c)
            self.h_t_ai = RNN(aigru, return_sequences=True)(point_domain)
        elif ai == "Zonotope":
            zonotope_domain = Zonotope(delta, fi)(look_up_c)
            self.h_t_ai = RNN(aigru, return_sequences=True)(zonotope_domain)
        else:
            raise NotImplementedError()

        weights, bias = self.squeeze.weights
        to_AI = lambda d: (lambda x: AI(x[:, :, :d], x[:, :, d:d * 2], x[:, :, d * 2:], False))
        attention = Lambda(lambda x: to_AI(units)(x).matmul(weights).bias_add(bias).softmax(
            axis=1).to_state())(self.h_t_ai)
        attention_out = Lambda(lambda x: to_AI(units)(x).portion_sum(to_AI(1)(attention)).to_state())(self.h_t_ai)
        weights, bias = self.fc1.weights
        to_AI1 = lambda d: (lambda x: AI(x[:, :d], x[:, d:d * 2], x[:, d * 2:], False))
        out = Lambda(lambda x: to_AI1(units)(x).matmul(weights).bias_add(bias).to_state())(attention_out)
        self.verify_model = Model(inputs=self.c, outputs=out)


params = {
    "D": 64,
    "batch_size": 32,
    "max_len": 300,
    "log_name_rep_init_scale": -1,
    "log_hidden_init_scale": -1,
    "conv_layer2_nfilters": 32,
    "log_lr": -3,
}

# model = SimpleRNNModel(params, 60, 4)
