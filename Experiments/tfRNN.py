from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

#First step here should be to take the time series data, and convert it to two arrays as input for training, the array for time step t and t+1.

class sortData():

	def __init__(self,minIdx = 1,maxIdx = 100):
		self.inputs = []
		self.outputs = []
		itrIdx = minIdx
		while itrIdx < maxIdx:
			temp = np.loadtxt("./Spike Results/1idTimes.csv", delimiter = ',' ).transpose()
			for i in range(len(temp)-1):
				self.inputs.append(temp[i])
				self.outputs.append(temp[i+1])
				itrIdx +=1
		self.batchID = 0
		popLen = len(temp)

	def next(self,batchSize):
		if self.batchId == len(self.data):
			self.batchId = 0
		batchData = self.data[self.batchId:min(self.batchId + batchSize, len(self.data))]
		batchLabels = self.labels[self.batchId:min(self.batchId + batchSize, len(self.data))]
		self.batchId = min(self.batchId + batchSize, len(self.data))
		return batchData, batchLabels

# Network Parameters
batchSize = 6
numOutput = 1
numInput = 1
numHidden = 1
learningRate = .01
trainingSteps = 100
popLen = 10
epochLen = 20

training = sortData(1,75)
testing = sortData(75,100)

# tf Graph input
X = tf.placeholder("float", [None, popLen, numInput])
Y = tf.placeholder("float", [None, popLen, numOutput])

# Define weights
weights = {'out': tf.Variable(tf.random_normal([numHidden, numClasses]))}
biases = {'out': tf.Variable(tf.random_normal([numClasses]))}


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(numHidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y = training.next(batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    test_len = 128
    test_data = testing.data
    test_label = testing.labels
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
