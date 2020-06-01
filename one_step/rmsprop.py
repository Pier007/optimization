# -*- coding: utf-8 -*-

import math
from step_search import Rosenbrock
from step_search import get_grad

# RMSProp 方法
def rmsprop(a, b, init_point, eps):

	gama = 0.9
	alpha = 0.003
	e = 1e-8
	s = [0, 0]

	x = init_point[0]
	y = init_point[1]

	pre = 0
	now = Rosenbrock(a, b, x, y)
	n = 0

	while math.fabs(now-pre) > eps:
		grad = get_grad(a, b, x, y)
		for i in range(len(s)):
			s[i] = gama*s[i] + (1-gama)*grad[i]**2
		x = x - alpha/(e+math.sqrt(s[0]))*grad[0]
		y = y - alpha/(e+math.sqrt(s[1]))*grad[1]
		print("point: " + str(x) + ', ' + str(y))
		pre = now
		now = Rosenbrock(a, b, x, y)
		print("function value: " + str(now))
		n = n + 1

	min_point = [x, y]
	print("total times: " + str(n))

	return min_point

if __name__ == '__main__':

	print(rmsprop(1, 100, [10,10], 0.000001))