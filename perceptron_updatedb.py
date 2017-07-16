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

#TODO: Figure out how to make this work with n-dimensional inputs. Basically just
#need to make W an n-dimensional tf.Variable and then figure out how to do a dot
#product with pts
#TODO: Modify algorithm to handle intercept (b), as in wiki article
"""
Takes in (x,y) pairs, and their classification, and returns the weights of the 
linear classifier that separates them. Assumes that the points are seperable.
INPUTS: 
x_train: x coords of training points
y_train: y coords of training points
sgn: {1,-1} vector containing classification of each (x,y) pair
RETURNS: Parameters of fitted linear model
"""
def main(x_train,y_train,sgn):
	
	#DO NOT INITIALIZE ANYTHING TO 0
	W1 = tf.Variable([0.1])
	W2 = tf.Variable([0.1])
	b = tf.Variable([1.0]) #Initialized to 1 
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

	#Perceptron algorithm. Repeatedly finds a misclassified point and updates weights
	#until all points are correctly classified
	while(True):
		i = misclassified(x_train,y_train,sgn)
		if i == -1:
			break
		#Update step. Tensorflow automatically updates values
		oldW1 = sess.run(W1)[0]
		oldW2 = sess.run(W2)[0]
		sess.run(tf.assign(W1, [oldW1 + x_train[i] * sgn[i]]))
		sess.run(tf.assign(W2, [oldW2 + y_train[i] * sgn[i]]))
		print sess.run(W1), sess.run(W2)

	return sess.run(W1)[0], sess.run(W2)[0]
		
if __name__ == '__main__':
	#Test set: Used points that didn't have an intercept at 0. Example below is y = 2x+1
	#Uses strict inequality (.5 margin) for simplicity
	x_train = [0,1,2,3,4,5,6,7]
	y_train = [1,3,5,7,9,11,13,15]
	pts = zip(x_train, y_train)
	sgn = [1, 1, 1, -1, 1, -1, 1, 1, 1, 1] # + = 1, - = -1

	W1, W2 = main(x_train, y_train, sgn)
