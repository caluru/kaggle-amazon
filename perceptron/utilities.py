import random
import numpy as np

"""
Generates a linearly seperable set of points
@param func: The target seperator (function should take in 
	one argument, a list with values for each variable
@param nVars: Number of variables taken in by the function
@param nPoints: The number of points to generate
@return points: The points that were generated
@return classes: Classifications of each generated point 
	according to the function provided
"""
def generatePoints(func, nVars, nPoints):
	points = []
	classes = []
	for i in range(nPoints):
		sgn = 0
		while(sgn == 0):
			temp = [random.randint(-100,100) for _ in range(nVars)]
			sgn = np.sign(func(temp))
		points.append(temp)
		classes.append(sgn)
	return points, classes
