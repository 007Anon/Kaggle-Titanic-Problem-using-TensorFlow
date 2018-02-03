import plot
import tensorflow as tf
import numpy as np
from pandas import read_csv, DataFrame
# import train_nn

print("Loading Data....")
data = read_csv("train.csv")
labels = read_csv("labels.csv")

test = read_csv("test.csv")
test_labels = read_csv("test_labels.csv")

cross_val = read_csv("cross_val.csv")
cross_val_labels = read_csv("cross_val_labels.csv")

to_predict = read_csv("predict.csv")
theta = np.loadtxt("theta.txt").reshape(data.shape[1], 1)
print("100%")


m, n = data.shape
m_test, _ = test.shape
m_cross_val = cross_val.shape


# todo.NEW Train_nn.py


def train_nn(x, y, test, test_labels, cross_val, cross_val_labels, max_iter=1000, hidden_nodes=5, alpha=0.001, batch_size = None):
    input_nodes = x.shape[1]
    output_nodes = y.shape[1]

    batch_size = x.shape[0] if not batch_size else batch_size

    def predict(x, weights, biases, keep_prob=1.0):
        layer1 = tf.add(tf.matmul(x, weights['layer1']), biases['layer1'])
        layer1 = tf.nn.relu(layer1)
        layer1 = tf.nn.dropout(layer1, keep_prob)
        output = tf.matmul(layer1, weights['output']) + biases['output']
        return output

    weights = {
        'layer1': tf.Variable(tf.random_normal([input_nodes, hidden_nodes])),
        'output': tf.Variable(tf.random_normal([hidden_nodes, output_nodes]))
    }

    biases = {
        'layer1': tf.Variable(tf.random_normal([hidden_nodes])),
        'output': tf.Variable(tf.random_normal([output_nodes]))
    }
    keep_prob = tf.placeholder("float")

    X = tf.placeholder('float', [None, x.shape[1]])
    Y = tf.placeholder('float', [None, output_nodes])
    predictions = predict(X, weights, biases, keep_prob)
    j = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha).minimize(j)

    j_history = []
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        for i in range(max_iter):
            j_avg = 0.0
            batches = int(len(x)/batch_size)
            x_batches = np.array_split(x, batches)
            y_batches = np.array_split(y, batches)

            for batch in range(batches):
                x1, y1 = x_batches[batch], y_batches[batch]
                _, c = sess.run([optimizer, j], feed_dict={X: x1, Y: y1, keep_prob: 0.8})

                j_avg += c/batches
                j_history.append(j_avg)
        print("\r" + "{}%".format(int(100 * i / max_iter) + 1), end="") if not i % (max_iter / 100) else print("",
                                                                                                               end="")
        print("Optimization finished successfully!")

        weights = sess.run(weights)
        biases = sess.run(biases)
        train_prediction = sess.run(predict(x, weights, biases))
        test_prediction = sess.run(predict(test, weights, biases))
        cross_val_prediction = sess.run(predict(cross_val, weights, biases))

        # train_accuracy = (np.sum(DataFrame(train_prediction).values == y.values) / m) * 100
        # print(train_accuracy)
        # test_accuracy = (np.sum(DataFrame(test_prediction).values == test_labels.values) / m_test) * 100
        # cross_val_accuracy = (np.sum(DataFrame(cross_val_prediction).values == cross_val_labels.values) / m_cross_val) * 100
        # gross_accuracy = (train_accuracy + test_accuracy + cross_val_accuracy) / 3


        # print("*******************************************")
        # print("Accuracy on training set: {0: .2f} %".format(train_accuracy))
        # print("Accuracy on test set: {0: .2f} %".format(test_accuracy))
        # print("Accuracy on cross validation set: {0: .2f} %".format(cross_val_accuracy))
        # print("Gross Accuracy: {0: .2f} %".format(gross_accuracy))
        # print("*******************************************")


        np.savetxt("layer1_weights.csv", weights['layer1'], delimiter=",")
        np.savetxt("layer1_bias.csv", biases['layer1'], delimiter=",")
        np.savetxt("output_weights.csv", weights['output'], delimiter=",")
        np.savetxt("output_bias.csv", biases['output'], delimiter=",")

    plot.plot(j_history, max_iter)






x1 = data.astype(np.float32)
y1 = labels.astype(np.float32)
test = test.astype(np.float32)
test_labels = test_labels.astype(np.float32)
cross_val = cross_val.astype(np.float32)
cross_val_labels = cross_val_labels.astype(np.float32)
train_nn(x1, y1, test, test_labels, cross_val, cross_val_labels, hidden_nodes=10)
