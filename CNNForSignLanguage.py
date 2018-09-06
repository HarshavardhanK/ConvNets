#Tutorials

import tensorflow as tf

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from subprocess import check_output

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

from sklearn.model_selection import train_test_split

import glob
import os
import math
import operator
import functools
import random
import datetime

from scipy.ndimage.interpolation import rotate, shift, zoom

import time

data_file_X_path = "/Users/HarshavardhanK/Google Drive/Code/Datasets/SignLanguage/Sign-language-digits-dataset 2/X.npy"
data_file_Y_path = "/Users/HarshavardhanK/Google Drive/Code/Datasets/SignLanguage/Sign-language-digits-dataset 2/Y.npy"

def load_data(path_X, path_Y):

    X = np.load(path_X)
    Y = np.load(path_Y)

    test_size = 0.15

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)

    print('Training shape:', X_train.shape)
    print(X_train.shape[0], 'sample', X_train.shape[1], 'x', X_train.shape[2], 'size grayscale image.')
    print('Test shape', X_test.shape)
    print(X_test.shape[0], 'sample', X_test.shape[1], 'x', X_test.shape[2], 'size grayscale image.')

    Y_test_cls = np.argmax(Y_test, axis=1)
    Y_train_cls = np.argmax(Y_train, axis=1)

    return (X_train, X_test, Y_train, Y_test, Y_train_cls, Y_test_cls)

X_train, X_test, Y_train, Y_test, Y_train_cls, Y_test_cls = load_data(data_file_X_path, data_file_Y_path)

# DATA DIMENSIONS

start_time = time.time()
print(start_time)


img_size = 64
img_size_flat = img_size ** 2
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10

def plot_images(images, cls_true, cls_pred=None):
    assert(len(images) == len(cls_true) == 9)

    #Create figure with 3x3 subplots

    fig, axes = plt.subplots(3,3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        #Plot images
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        #Show true and predicted classes

        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])


    #Display the plot
    plt.show()

def plot_some_smaples():

    images = X_test[0:9]
    cls_true = Y_test_cls[0:9]

    plot_images(images=images, cls_true=cls_true)

#plot_some_smaples()

learning_rate = 0.001
epochs = 40000
batch_size = 16
display_step = 20

n_input = img_size_flat
dropout = 0.75

#Building TF graph

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

print('Shape of placeholder', x.shape, y.shape)

train_X = X_train
train_Y = Y_train
new_train_X = train_X.reshape(X_train.shape[0],img_size_flat)
new_test_X = X_test.reshape(X_test.shape[0],img_size_flat)

def conv2D(x, W, b, strides=1):

    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)

    #print('conv2D')

    return tf.nn.relu(x)

def maxpool2D(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def conv_net(x, weights, biases, dropout):
    #reshape input to 64x64 size
    x = tf.reshape(x, shape=[-1, 64, 64, 1])

    #Convolutional layer 1
    conv1 = conv2D(x, weights['wc1'], biases['bc1'])

    #Max pooling
    conv1 = maxpool2D(conv1, k=2)

    #Convolutional layer 2
    conv2 = conv2D(conv1, weights['wc2'], biases['bc2'])

    #Max pooling
    conv2 = maxpool2D(conv2, k=2)

    #FC layers
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])

    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return out


weights = {
'wc1': tf.Variable(tf.random_normal([5,5,1,32]), name='wc1'),
'wc2': tf.Variable(tf.random_normal([5,5,32,64]), name='wc2'),
'wd1': tf.Variable(tf.random_normal([64 * 64 * 4, 1024]), name='wd1'),
'out': tf.Variable(tf.random_normal([1024, num_classes]), name='out')
}

biases = {
'bc1': tf.Variable(tf.random_normal([32]),name='bc1'),
'bc2': tf.Variable(tf.random_normal([64]),name='bc2'),
'bd1': tf.Variable(tf.random_normal([1024]),name='bd1'),
'out': tf.Variable(tf.random_normal([num_classes]),name='bout')
}

model = conv_net(x, weights, biases, keep_prob)
print(model)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

y_true_cls = tf.argmax(y, 1)
y_pred_cls = tf.argmax(y, 1)

correct_model = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_model, tf.float32))

init = tf.initializers.global_variables()

def get_batch(X, Y, batchSize=16):
    length = X.shape[0]
    count = 0

    while count < length / batchSize:
        random.seed(datetime.datetime.now())
        randstart = random.randint(0, length - batchSize - 1)
        count += 1

        yield (X[randstart:randstart+batchSize], Y[randstart:randstart+batchSize])

loss_t = []
steps_t = []
acc_t = []

with tf.Session() as sess:

    sess.run(init)
    step = 1

    while step * batch_size < epochs:

        a = get_batch(new_test_X, train_Y, batch_size)
        batch_x, batch_y = next(a)

        sess.run(optimizer, feed_dict={x: batch_x, y:batch_y, keep_prob:dropout})

        if step % display_step == 0:
            print('*'*40)

            loss, acc = sess.run([cost, accuracy], feed_dict={x:batch_x, y:batch_y, keep_prob:1.})

            print("Iter " + str(step*batch_size) + ", Loss= " + "{:.3f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))
            loss_t.append(loss)
            steps_t.append(step*batch_size)
            acc_t.append(acc)

        step += 1

    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: new_test_X,  y: Y_test,  keep_prob: 1.}))

    cls_pred = sess.run(y_pred_cls, feed_dict={x: new_test_X, y: Y_test, keep_prob: 1.})

print("--- %s seconds ---" % (time.time() - start_time))

plt.plot(steps_t, loss_t, 'r--')
plt.xlabel("Number of iterarion")
plt.ylabel("Loss")
plt.show()
