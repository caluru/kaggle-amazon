"""
Toy example implementing the (2D) perceptron algorithm (https://en.wikipedia.org/wiki/Perceptron)
using Google's TensorFlow API. 
"""

import tensorflow as tf
import numpy as np

#Shuffles 3 arrays in tandem
def shuffle(a,b,c):
	rng_state = np.random.get_state()
	np.random.shuffle(a)
	np.random.set_state(rng_state)
	np.random.shuffle(b)
	np.random.set_state(rng_state)
	np.random.shuffle(c)

#Takes in (x,y) pairs, and their classification, and returns the weights of the 
#linear classifier that separates them. Assumes that the points are seperable.
def main(x_train,y_train,sgn):
	
	W1 = tf.Variable([0.1])
	W2 = tf.Variable([0.1])
	b = tf.Variable([0.0])
	x = tf.placeholder(tf.float32)
	y = tf.placeholder(tf.float32)

	linear_model = W1 * x + W2 * y + b
	classifier = tf.sign(linear_model)

	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)

	#Returns the index of the first misclassified point, or -1 if there are none
	def misclassified(x_train,y_train,sgn):
		shuffle(x_train,y_train,sgn) #not strictly necessary, but may help with convergence?
		outputs = sess.run(classifier, {x:x_train, y:y_train})
		for i in range(len(outputs)):
			if outputs[i] != sgn[i] and outputs[i] != 0:
				return i
		return -1

	while(True):
		i = misclassified(x_train,y_train,sgn)
		if i == -1:
			break
		#Update step. This is wrong
		oldW1 = sess.run(W1)[0]
		oldW2 = sess.run(W2)[0]
		sess.run(tf.assign(W1, [oldW1 + x_train[i] * sgn[i]]))
		sess.run(tf.assign(W2, [oldW2 + y_train[i] * sgn[i]]))
		print sess.run(W1), sess.run(W2)

	return W1, W2
		
if __name__ == '__main__':
	#Test set, with positive examples being above the line y = x / 2
	#Uses strict inequality (.5 margin) for simplicity
	x_train = [-2, -3, 3, 6, 0, 0, 2, 3, 4, -5]
	y_train = [0, -1, 2, 4, 1, -1, 0, -2, 1, -3]
	points = zip(x_train, y_train)
	sgn = [1, 1, 1, 1, 1, -1, -1, -1, -1, -1] # + = 1, - = -1

	W1, W2 = main(x_train, y_train, sgn)
"""
	W = tf.Variable([1.0])
	b = tf.Variable([0.0])
	x = tf.placeholder(tf.float32)
	y = tf.placeholder(tf.float32)

	linear_model = W * x + b

	sess = tf.Session()
	init = tf.global_variables_initializer()
	sess.run(init)

	sess.run(linear_model, {x:[1,2,3]})
"""

