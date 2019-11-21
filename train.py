import tensorflow as tf
import numpy as np

from rnn_verification.simple_rnn_model import SimpleRNNModel, params, SimpleRNNModel1
from logic_formula.utils import get_one_hot

training_X = np.load("./AG/X_train.npy")
training_y = np.load("./AG/y_train.npy")
training_num = len(training_X)
# shuffle_ids = np.array(np.arange(training_num))
# np.random.shuffle(shuffle_ids)
# training_X = training_X[shuffle_ids]
# training_y = training_y[shuffle_ids]
test_X = np.load("./AG/X_test.npy")
test_y = np.load("./AG/y_test.npy")
nb_classes = 4
training_Y = get_one_hot(training_y, nb_classes)
test_Y = get_one_hot(test_y, nb_classes)

model = SimpleRNNModel1(params, 60, nb_classes)
model.model.fit(x=training_X, y=training_Y, batch_size=64, epochs=30, callbacks=[model.early_stopping], verbose=2,
                validation_data=(test_X[:500], test_Y[:500]), shuffle=True)
model.model.save_weights(filepath="./models/rnn")
# batch_size = 64
# params["batch_size"] = batch_size
# model = SimpleRNNModel(params, 60, 4)
# saver = tf.train.Saver()
# epochs = 30
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
# best = 1e10
# last_best = 0
# patience = 5
# # merged = tf.summary.merge_all()
# # train_writer = tf.summary.FileWriter('./train/rnn_attention_infer', sess.graph)
# for iter in range(epochs):
#     for i in range(1, (training_num - 1) // batch_size + 2):
#         ids = range((i - 1) * batch_size, min(i * batch_size, training_num))
#         batch_X = training_X[ids]
#         batch_Y = training_Y[ids]
#
#         sess.run(model.optimizer, feed_dict={model.c: batch_X, model.y: batch_Y})
#         if i % 100 == 0:
#             print("Processing %.2f%%" % (i * batch_size * 100.0 / training_num))
#     acc, losses = sess.run([model.accuracy, model.loss],
#                            feed_dict={model.c: test_X[:500], model.y: test_Y[:500]})
#     if losses < best:
#         best = losses
#         last_best = iter
#     else:
#         if iter - last_best > patience:
#             print("early stoppint at iter %d" % iter)
#             break
#     # train_writer.add_summary(summary, iter)
#     print("Test accuracy: %.2f%%\tTest loss: %g" % (acc * 100, losses))
#
# save_path = saver.save(sess, "./models/rnn_attention.ckpt")
# acc, losses = sess.run([model.accuracy, model.loss], feed_dict={model.c: test_X, model.y: test_Y})
# print("Test accuracy: %.2f%%\tTest loss: %g" % (acc * 100, losses))
