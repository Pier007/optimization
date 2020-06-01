# -*- coding: utf-8 -*-

import math
from step_search import Rosenbrock
from step_search import get_grad

# AdaGrad 方法
def adagrad(a, b, init_point, eps):

	alpha = 0.2
	e = 1e-8
	s = [0, 0]

	x = init_point[0]
	y = init_point[1]

	pre_pre = 0
	pre = 0
	now = Rosenbrock(a, b, x, y)
	n = 0

	while math.fabs(now-pre) > eps and math.fabs(now-pre_pre) > eps:
		grad = get_grad(a, b, x, y)
		for i in range(len(s)):
			s[i] = s[i] + grad[i]**2
		x = x - alpha/(e+math.sqrt(s[0]))*grad[0]
		y = y - alpha/(e+math.sqrt(s[1]))*grad[1]
		print("point: " + str(x) + ', ' + str(y))
		# 处理震荡情况
		pre_pre = pre
		pre = now
		now = Rosenbrock(a, b, x, y)
		print("function value: " + str(now))
		n = n + 1

	min_point = [x, y]
	print("total times: " + str(n))

	return min_point

if __name__ == '__main__':

	print(adagrad(1, 100, [10,10], 0.000001))