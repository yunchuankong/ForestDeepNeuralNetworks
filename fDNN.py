from __future__ import print_function
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from random import seed
import sys
import time

tf.reset_default_graph()
file = sys.argv[1]
expression = np.loadtxt(file, dtype=float, delimiter=",", skiprows=1)
label_vec = np.array(expression[:,-1], dtype=int)
expression = np.array(expression[:,:-1])

cutpoint = 500
expression, label_vec = shuffle(expression, label_vec) ## different here from rfnn.py
expression_train = expression[:cutpoint, :]
expression_test = expression[cutpoint:, :]
y_train = label_vec[:cutpoint]
y_test = label_vec[cutpoint:]

## RF part
n_trees = 300
rf = RandomForestClassifier(n_estimators=n_trees, n_jobs=-1)
rf.fit(expression_train, y_train)

x_total = [tree.predict(expression) for tree in rf.estimators_]
x_total = np.transpose(x_total)
# print(np.shape(x_total))

## one-hot encode the binary inputs from forests
x_train = []
for sample in x_total:
    s = []
    for feature in sample:
        if feature == 1:
            s.append([0, 1])
        else:
            s.append([1, 0])
    x_train.append(s)
x_train = np.array(x_train)
x_test = x_train[cutpoint:, :] ## order is fixed from the forest, do not shuffle
x_train = x_train[:cutpoint, :]

## one-hot encode the labels
labels = []
for l in label_vec:
    if l == 1:
        labels.append([0, 1])
    else:
        labels.append([1, 0])
labels = np.array(labels,dtype=int)

y_train = labels[:cutpoint, :]
y_test = labels[cutpoint:, :]

## hyper-parameters and settings
L2 = True
droph1 = False
learning_rate = 0.0001
training_epochs = 200
batch_size = 8
display_step = 1

n_hidden_1 = 256
n_hidden_2 = 64
n_hidden_3 = 16
# n_hidden_4 = 16
n_classes = 2
n_features = np.shape(x_train)[1]

## initiate training logs
loss_rec = np.zeros([training_epochs, 1])
training_eval = np.zeros([training_epochs, 2])
# testing_eval = np.zeros([int(training_epochs/10), 2])
avg_test_acc = 0.
avg_test_auc = 0.

def multilayer_perceptron(x, weights, biases, keep_prob):

    # layer_1 = tf.add(tf.matmul(x, tf.multiply(weights['h1'], partition)), biases['b1'])
    # layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.add(tf.tensordot(x, weights['h1'], axes=[[1, 2],[0, 1]]), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # layer_1 = tf.nn.relu(layer_1)
    if droph1:
        layer_1 = tf.nn.dropout(layer_1, keep_prob=keep_prob)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2, keep_prob=keep_prob)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    layer_3 = tf.nn.dropout(layer_3, keep_prob=keep_prob)

    # layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    # layer_4 = tf.nn.relu(layer_4)
    # layer_4 = tf.nn.dropout(layer_4, keep_prob=keep_prob)

    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer


x = tf.placeholder(tf.float32, [None, n_features, 2])
y = tf.placeholder(tf.int32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)
lr = tf.placeholder(tf.float32)

weights = {
    'h1': tf.Variable(tf.truncated_normal(shape=[2, n_features, n_hidden_1], stddev=0.1)),
    'h2': tf.Variable(tf.truncated_normal(shape=[n_hidden_1, n_hidden_2], stddev=0.1)),
    'h3': tf.Variable(tf.truncated_normal(shape=[n_hidden_2, n_hidden_3], stddev=0.1)),
    # 'h4': tf.Variable(tf.truncated_normal(shape=[n_hidden_3, n_hidden_4], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal(shape=[n_hidden_3, n_classes], stddev=0.1))

}

biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'b2': tf.Variable(tf.zeros([n_hidden_2])),
    'b3': tf.Variable(tf.zeros([n_hidden_3])),
    # 'b4': tf.Variable(tf.zeros([n_hidden_4])),
    'out': tf.Variable(tf.zeros([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
if L2:
    reg = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2']) + \
          tf.nn.l2_loss(weights['h3']) + tf.nn.l2_loss(weights['out'])
    cost = tf.reduce_mean(cost + 0.1 * reg)
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

## Evaluation
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
y_score = tf.nn.softmax(logits=pred)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    total_batch = int(np.shape(x_train)[0] / batch_size)

    ## for monitoring weights
    # w1_pre = sess.run(weights['h1'][:10, :10], feed_dict={x: expression, y: labels, keep_prob: 1})

    ## Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        x_tmp, y_tmp = shuffle(x_train, y_train)
        # Loop over all batches
        for i in range(total_batch-1):
            batch_x, batch_y = x_tmp[i*batch_size:i*batch_size+batch_size], \
                                y_tmp[i*batch_size:i*batch_size+batch_size]
            _, c= sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y,
                                                        keep_prob: 0.9,
                                                        lr: learning_rate
                                                        })
            # Compute average loss
            avg_cost += c / total_batch

        del x_tmp
        del y_tmp

        ## Display logs per epoch step
        if epoch % display_step == 0:
            loss_rec[epoch] = avg_cost
            acc, y_s = sess.run([accuracy, y_score], feed_dict={x: x_train, y: y_train, keep_prob: 1})
            auc = metrics.roc_auc_score(y_train, y_s)
            training_eval[epoch] = [acc, auc]
            print ("Epoch:", '%d' % (epoch+1), "cost =", "{:.9f}".format(avg_cost),
                    "Training accuracy:", round(acc,3), " Training auc:", round(auc,3))
        if avg_cost <= 0.01:
            print("Early stopping.")
            break

    ## Testing cycle
    acc, y_s = sess.run([accuracy, y_score], feed_dict={x: x_test, y: y_test, keep_prob: 1})
    auc = metrics.roc_auc_score(y_test, y_s)
    print("*****=====", "Testing accuracy: ", acc, " Testing auc: ", auc, "=====*****")

