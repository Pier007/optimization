# -*- coding: utf-8 -*-

import math
from step_search import Rosenbrock
from step_search import get_grad

# 牛顿动量法
def nesterov(a, b, init_point, eps):

	# 固定参数 α 和 β
	beta = 0.99
	alpha = 0.002

	x = init_point[0]
	y = init_point[1]

	pre = 0
	now = Rosenbrock(a, b, x, y)
	n = 1

	# x(0) 的下降方向即负梯度方向
	grad = get_grad(a, b, x, y)
	v = grad[:]
	for i in range(len(v)):
		v[i] = -v[i]
	x = x + alpha*v[0]
	y = y + alpha*v[1]
	v_pre = v

	while math.fabs(now-pre) > eps:
		grad = get_grad(a, b, x, y)
		for i in range(len(v)):
			v[i] = beta*v_pre[i] - alpha*grad[i]
		x = x + v[0]
		y = y + v[1]
		print("point: " + str(x) + ', ' + str(y))
		v_pre = v
		pre = now
		now = Rosenbrock(a, b, x, y)
		print("function value: " + str(now))
		n = n + 1
		# β 衰减
		if n % 1 == 0:
			beta = beta * 0.99

	min_point = [x, y]
	print("total times: " + str(n))

	return min_point

if __name__ == '__main__':

	print(nesterov(1, 100, [10,10], 0.000001))