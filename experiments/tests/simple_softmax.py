import tensorflow.compat.v1 as tf
import numpy as np

tf.compat.v1.disable_v2_behavior()

x = tf.placeholder("float64", [None, 3])
y = tf.placeholder("float64", [None, 1])
W = tf.Variable(np.array([[0.1],[0.2],[0.3]]))
b = tf.Variable(np.array([0.8]))


y_ = tf.matmul(x,W) + b
cost = tf.reduce_mean(tf.reduce_mean((y-y_)*(y-y_), reduction_indices=1))

learning_rate = 1e-2
optim = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()

data = np.array([
  [1.0,2.0,3.0],
  [4.0,5.0,6.0],
  [6.0,7.0,8.0]
])

label = []
for i in range(3):
  label.append([0.42*data[i][0] + 0.35*data[i][1] + 0.53*data[i][2] + 5.0])

label = np.array(label)
print(label)

with tf.Session() as sess:
  sess.run(init)
  for i in range(4):
    _, loss = sess.run([optim, cost], feed_dict={x: data, y: label})
    print(loss)
