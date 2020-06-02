# -*- coding: utf-8 -*-

import math
import numpy as np
from step_search import Rosenbrock
from step_search import get_grad
from step_search import get_hessian

# 牛顿法
def newton(a, b, init_point, eps):

	x = init_point[0]
	y = init_point[1]

	pre_pre = 0
	pre = 0
	now = Rosenbrock(a, b, x, y)
	n = 0

	while math.fabs(now-pre) > eps and math.fabs(now-pre_pre) > eps:
		hessian = np.array(get_hessian(b, x, y))
		hessian_ = np.linalg.inv(hessian)
		grad = np.array(get_grad(a, b, x, y))
		x = x - np.dot(hessian_[0], grad)
		y = y - np.dot(hessian_[1], grad)
		print("point: " + str(x) + ', ' + str(y))
		pre_pre = pre
		pre = now
		now = Rosenbrock(a, b, x, y)
		print("function value: " + str(now))
		n = n + 1

	min_point = [x, y]
	print("total times: " + str(n))

	return min_point

if __name__ == '__main__':

	print(newton(1, 100, [3,3], 0.000001))

