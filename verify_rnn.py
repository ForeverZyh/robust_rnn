import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import numpy as np

from rnn_verification.simple_rnn_model import params, SimpleRNNModel, SimpleRNNModel1
from logic_formula.utils import get_one_hot

nb_classes = 4
# model = SimpleRNNModel(params, 60, nb_classes)
# model.model.load_weights("./models/rnn_attention")
#
# model.verify_swap(delta=1, ai="Box")

model1 = SimpleRNNModel1(params, 60, nb_classes)
model1.model.load_weights("./models/rnn")

model1.verify_swap(delta=1, ai="Zonotope", dp=True)
sess = tf.Session()
# graph = tf.compat.v1.get_default_graph()
# print(graph.get_operation_by_name("rnn_1/while/add_2"))
# train_writer = tf.summary.FileWriter('./train_box', sess.graph)

test_X = np.load("./AG/X_test.npy")
test_y = np.load("./AG/y_test.npy")
nb_classes = 4
test_Y = get_one_hot(test_y, nb_classes)

print(model1.verify_model.predict(x=test_X[:1, :params["max_len"]]))
# print(model.verify_model.predict(x=test_X[:2]))
