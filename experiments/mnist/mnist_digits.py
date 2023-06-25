
import tensorflow as tf2
import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
print ("Packs loaded")
(trainimg, trainlabel), (testimg, testlabel) = tf2.keras.datasets.mnist.load_data()
trainimg = np.reshape(trainimg, (-1, 784)).astype(float)
testimg = np.reshape(testimg, (-1, 784)).astype(float)
trainimg /= 255.0
testimg /= 255.0
trainlabel = np.eye(np.max(trainlabel) + 1)[trainlabel]
testlabel = np.eye(np.max(testlabel) + 1)[testlabel]
print ("MNIST loaded")

tf.compat.v1.disable_v2_behavior()

# Parameters of logistic regression
learning_rate = 0.01
training_epochs = 50
batch_size = 100
display_step = 1

#Create Graph for Logistic Regression
x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#Activation, Cost,Optimizing functions

actv = tf.nn.softmax(tf.matmul(x,W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(actv), reduction_indices=1))

optim = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#Gradient Descent

pred = tf.equal(tf.argmax(actv, 1), tf.argmax(y,1))
accr = tf.reduce_mean(tf.cast(pred, "float"))
#Optimize with tensorflow
#Initializing the variables
init = tf.initialize_all_variables()
print ("Network constructed")

#Launch the graph
with tf.Session() as sess:
	sess.run(init)

	#Training cycle
	for epoch in range(training_epochs):
		avg_cost = 0.
		num_batch = int(trainimg.shape[0]/batch_size)
		#Loop over all batches
		for i in range(num_batch):
			if 0: #Using tensorflow API
				batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			else: # Random batch sampling
				randidx = np.random.randint(trainimg.shape[0], size=batch_size)
				batch_xs = trainimg[randidx, :]
				batch_ys = trainlabel[randidx, :]

			#Fit training using batch data
			sess.run(optim, feed_dict={x: batch_xs, y: batch_ys})
			#Compute average loss
			avg_cost += sess.run(cost, feed_dict={x:batch_xs, y:batch_ys})/num_batch

		#Display logs per epoch step
		if epoch%display_step ==0:
			train_acc = accr.eval({x:batch_xs, y:batch_ys})
			print ("Epoch: %03d/%03d cost: %.9f train_acc: %.3f" % (epoch, training_epochs, avg_cost, train_acc))

	print ("Optimization Complete!")

	#Test model
	#Calculate accuracy
	print ("Accuracy: ", accr.eval({x: testimg, y: testlabel}))

print("Done.")