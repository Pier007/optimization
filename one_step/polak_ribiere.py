# -*- coding: utf-8 -*-

import math
from step_search import Rosenbrock
from step_search import get_grad
from step_search import get_step

# 共轭梯度法变种之 Polak Ribiere 方法
def pr_method(a, b, init_point, interval, eps):

	x = init_point[0]
	y = init_point[1]

	pre = 0
	now = Rosenbrock(a, b, x, y)
	n = 2

	# x(0) 的下降方向即负梯度方向
	g_z = get_grad(a, b, x, y)
	for i in range(len(g_z)):
		g_z[i] = -g_z[i]
	step = get_step(a, b, x, y, interval, g_z)
	x = x + step*g_z[0]
	y = y + step*g_z[1]

	# x(1) 的下降方向即负梯度方向
	g_y = get_grad(a, b, x, y)
	for i in range(len(g_y)):
		g_y[i] = -g_y[i]
	d_y = g_y[:]
	step = get_step(a, b, x, y, interval, g_y)
	x = x + step*g_y[0]
	y = y + step*g_y[1]

	while math.fabs(now-pre) > eps:
		g_x = get_grad(a, b, x, y)
		# β 系数
		beta = (g_y[0]*(g_y[0]-g_z[0])+g_y[1]*(g_y[1]-g_z[1])) / (g_z[0]*g_z[0]+g_z[1]*g_z[1])
		if beta < 0:
			beta = 0

		d_x = [0, 0]
		for i in range(len(d_x)):
			d_x[i] = -g_x[i] + beta*d_y[i]
		# 归一化
		e = math.sqrt(math.pow(d_x[0],2) + math.pow(d_x[1],2))
		for i in range(len(d_x)):
			if e == 0:
				d_x[i] = 0
			else:
				d_x[i] = d_x[i] / e

		step = get_step(a, b, x, y, interval, d_x)
		x = x + step*d_x[0]
		y = y + step*d_x[1]
		print("point: " + str(x) + ', ' + str(y))
		d_y = d_x
		g_z = g_y
		g_y = g_x
		pre = now
		now = Rosenbrock(a, b, x, y)
		print("function value: " + str(now))
		n = n + 1

	min_point = [x, y]
	print("total times: " + str(n))

	return min_point

if __name__ == '__main__':

	print(pr_method(1, 100, [10,10], [0,0.005], 0.000001))