# -*- coding: utf-8 -*-

import math
from step_search import Rosenbrock
from step_search import get_grad

# Adam 方法
def adam(a, b, init_point, eps):

	alpha = 0.01
	gama_v = 0.9
	gama_s = 0.9
	e = 1e-8
	v = [0, 0]
	s = [0, 0]
	v_correct = [0, 0]
	s_correct = [0, 0]

	x = init_point[0]
	y = init_point[1]

	pre = 0
	now = Rosenbrock(a, b, x, y)
	n = 0

	while math.fabs(now-pre) > eps:
		grad = get_grad(a, b, x, y)
		if n == 0:
			v = grad[:]
		else:
			for i in range(len(v)):
				v[i] = gama_v*v[i] + (1-gama_v)*grad[i]
		for i in range(len(s)):
			s[i] = gama_s*s[i] + (1-gama_s)*grad[i]**2
			v_correct[i] = v[i] / (1-gama_v**(n+1))
			s_correct[i] = s[i] / (1-gama_s**(n+1))
		x = x - alpha*v_correct[0]/(e+math.sqrt(s_correct[0]))
		y = y - alpha*v_correct[1]/(e+math.sqrt(s_correct[1]))
		print("point: " + str(x) + ', ' + str(y))
		pre = now
		now = Rosenbrock(a, b, x, y)
		print("function value: " + str(now))
		n = n + 1

	min_point = [x, y]
	print("total times: " + str(n))

	return min_point

if __name__ == '__main__':

	print(adam(1, 100, [10,10], 0.000001))