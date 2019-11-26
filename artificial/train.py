import tensorflow as tf
import numpy as np

from simple_rnn_model import params, SimpleRNNModel1
from utils import get_one_hot

training_X = np.load("./X_train.npy")
training_y = np.load("./y_train.npy")
training_num = len(training_X)
test_X = np.load("./X_test.npy")
test_y = np.load("./y_test.npy")
nb_classes = 2
training_Y = get_one_hot(training_y, nb_classes)
test_Y = get_one_hot(test_y, nb_classes)
params["max_len"] = 100
params["D"] = 2
params["conv_layer2_nfilters"] = 3
model = SimpleRNNModel1(params, 30, nb_classes)
model.model.fit(x=training_X, y=training_Y, batch_size=64, epochs=10, callbacks=[model.early_stopping], verbose=2,
                validation_data=(test_X[:500], test_Y[:500]), shuffle=True)
model.model.save_weights(filepath="./models/rnn_tiny")
