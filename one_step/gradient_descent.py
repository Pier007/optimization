# -*- coding: utf-8 -*-

import math
from step_search import Rosenbrock
from step_search import get_grad
from step_search import get_step

# 梯度下降法
def gradient_descent(a, b, init_point, interval, eps):

	x = init_point[0]
	y = init_point[1]

	pre = 0
	now = Rosenbrock(a, b, x, y)
	n = 0

	while math.fabs(now-pre) > eps:
		grad = get_grad(a, b, x, y)
		# 取负梯度方向
		for i in range(len(grad)):
			grad[i] = -grad[i]
		step = get_step(a, b, x, y, interval, grad)
		x = x + step*grad[0]
		y = y + step*grad[1]
		print("point: " + str(x) + ', ' + str(y))
		pre = now
		now = Rosenbrock(a, b, x, y)
		print("function value: " + str(now))
		n = n + 1

	min_point = [x, y]
	print("total times: " + str(n))

	return min_point

if __name__ == '__main__':

	print(gradient_descent(1, 100, [10,10], [0,0.002], 0.000001))