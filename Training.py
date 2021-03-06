#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import time

start_time = time.time()
x = tf.placeholder('float')
y = tf.placeholder('float')
img_dimension = 50
num_slices = 20
num_classes = 2

processed_data = np.load('processed_data.npy')
training_set = processed_data[:-100]
validation_set = processed_data[-100:]

def maxpool3d(x):
    return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')

def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')

def cnn(x):
    weights = {'convolution1_weights':tf.Variable(tf.random_normal([3,3,3,1,32])), 'convolution2_weights':tf.Variable(tf.random_normal([5,5,5,32,64])),
               'fc_weights':tf.Variable(tf.random_normal([54080,1024])), 'out':tf.Variable(tf.random_normal([1024, num_classes]))}

    biases = {'convolution1_biases':tf.Variable(tf.random_normal([32])), 'convolution2_biases':tf.Variable(tf.random_normal([64])),
              'fc_biases':tf.Variable(tf.random_normal([1024])), 'out':tf.Variable(tf.random_normal([num_classes]))}

    x = tf.reshape(x, shape=[-1, img_dimension, img_dimension, num_slices, 1])

    convolution1 = tf.nn.relu(conv3d(x, weights['convolution1_weights']) + biases['convolution1_biases'])
    convolution1 = maxpool3d(convolution1)

    convolution2 = tf.nn.relu(conv3d(convolution1, weights['convolution2_weights']) + biases['convolution2_biases'])
    convolution2 = maxpool3d(convolution2)

    fc = tf.reshape(convolution2,[-1, 54080])
    fc = tf.nn.relu(tf.matmul(fc, weights['fc_weights'])+biases['fc_biases'])
    fc = tf.nn.dropout(fc, 0.8)

    output = tf.matmul(fc, weights['out'])+biases['out']
    return output

def training(x):
    new_time = time.time()
    prediction = cnn(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=.01).minimize(cost)

    num_epochs = 20
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        success_count = 0
        runs_count = 0

        for epoch in range(num_epochs):
            loss = 0
            new_time = time.time()
            for data in training_set:
                runs_count += 1
                try:
                    X = data[0]
                    Y = data[1]
                    _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
                    loss += c
                    success_count += 1
                except Exception as e:
                    print(str(e))
                    pass
            
            print("Completed epoch " + str(epoch+1) + " out of " + str(num_epochs))
            print("\tLoss: " + str(loss))
            print("\tTime to complete: %s seconds." % (time.time() - new_time))

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            accuracy_local = accuracy.eval({x:[i[0] for i in validation_set], y:[i[1] for i in validation_set]})
            print('\tAccuracy: ' + str(accuracy_local) + "\n")
            
        print("Jobs Done!")
        accuracy = accuracy.eval({x: [i[0] for i in validation_set], y: [i[1] for i in validation_set]})
        fitment = success_count/runs_count
        print("Finishing accuracy: " + str(accuracy))
        print("Finishing fitment: " + str(fitment))
        print("Total running time: %s seconds." % (time.time() - new_time))

        fh = open("output.txt", "a+")
        fh.write(str(accuracy) + "\n")
        fh.close()
        return(accuracy)

fh = open("output.txt", "a+")
fh.write("=== Beginning new Trial ===\n")
fh.close()
print("Beginning first cnn interation...")
accuracy = training(x)
while(accuracy < 0.69):
    print("\n === Accuracy is less than target, trying a new iteration... ===")
    accuracy = training(x)

print("Success!")
print("It took %s seconds to achieve an accuracy >= 0.69" % (time.time() - start_time))


# In[ ]:




