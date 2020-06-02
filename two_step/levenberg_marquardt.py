# -*- coding: utf-8 -*-

import math
import numpy as np
from step_search import Rosenbrock
from step_search import Rosenbrock_q
from step_search import get_grad
from step_search import get_hessian

# LM 方法
def lm(a, b, init_point, eps):

	e = 0.25
	B = np.array([[0,0],[0,0]])

	x = init_point[0]
	y = init_point[1]

	pre = 0
	now = Rosenbrock(a, b, x, y)
	pre_q = 0
	now_q = 0
	n = 0

	while math.fabs(now-pre) > eps:
		H = np.array(get_hessian(b, x, y))
		grad = np.array(get_grad(a, b, x, y))
		for i in range(len(H)):
			for j in range(len(H[0])):
				B[i][j] = e + H[i][j]
		B_ = np.linalg.inv(B)
		delta_x = -np.dot(B_[0], grad)
		delta_y = -np.dot(B_[1], grad)
		x = x + delta_x
		y = y + delta_y
		print("point: " + str(x) + ', ' + str(y))
		pre = now
		now = Rosenbrock(a, b, x, y)
		pre_q = now_q
		now_q = Rosenbrock_q(a, b, x, y, delta_x, delta_y, grad, H)
		print("function value: " + str(now))
		r = (now - pre) / (now_q - pre_q)
		if n > 2 and r > 0:
			if r < 0.25:
				e = 4*e
			elif r > 0.75:
				e = 0.5*e
		n = n + 1

	min_point = [x, y]
	print("total times: " + str(n))

	return min_point


if __name__ == '__main__':

	print(lm(1, 100, [3,3], 0.000001))