# -*- coding: utf-8 -*-

import math
from step_search import Rosenbrock
from step_search import get_grad

# AdaDelta 方法
def adadelta(a, b, init_point, eps):

	gama = 0.9
	e = 1e-8
	s1 = [0, 0]
	s2 = [0, 0]
	delta = [0, 0]

	x = init_point[0]
	y = init_point[1]
	pre_x = 0
	pre_y = 0

	pre = 0
	now = Rosenbrock(a, b, x, y)
	n = 0

	while math.fabs(now-pre) > eps:
		grad = get_grad(a, b, x, y)
		if n == 0:
			delta = grad[:]
		else:
			delta[0] = x - pre_x
			delta[1] = y - pre_y
		for i in range(len(s1)):
			s1[i] = gama*s1[i] + (1-gama)*delta[i]**2
		for i in range(len(s2)):
			s2[i] = gama*s2[i] + (1-gama)*grad[i]**2
		x = x - math.sqrt(s1[0])/(e+math.sqrt(s2[0]))*grad[0]
		y = y - math.sqrt(s1[1])/(e+math.sqrt(s2[1]))*grad[1]
		print("point: " + str(x) + ', ' + str(y))
		pre = now
		now = Rosenbrock(a, b, x, y)
		print("function value: " + str(now))
		pre_x = x
		pre_y = y
		n = n + 1

	min_point = [x, y]
	print("total times: " + str(n))

	return min_point

if __name__ == '__main__':

	print(adadelta(1, 100, [10,10], 0.000001))
