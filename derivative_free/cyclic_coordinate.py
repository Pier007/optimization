# -*- coding: utf-8 -*-

import math
from step_search import Rosenbrock
from step_search import get_step

# 坐标轮换法
def cyclic(a, b, init_point, interval, eps):

	x = init_point[0]
	y = init_point[1]
	direction = [[1,0],[0,1]]
	delta = [0, 0]

	pre = [0, 0, 0]
	now = [0, 0, 0]
	n = 0

	while True:
		if n % 2 == 0:
			# x 方向
			d = direction[0]
			step = get_step(a, b, x, y, interval, d)
			delta[0] = step*d[0]
			x = x + delta[0]
			print("point: " + str(x) + ', ' + str(y))
			pre[0] = now[0]
			now[0] = Rosenbrock(a, b, x, y)
			print("function value: " + str(now[0]))
		else:
			# y 方向
			d = direction[1]
			step = get_step(a, b, x, y, interval, d)
			delta[1] = step*d[1]
			y = y + delta[1]
			print("point: " + str(x) + ', ' + str(y))
			pre[1] = now[1]
			now[1] = Rosenbrock(a, b, x, y)
			print("function value: " + str(now[1]))
			# 加速步
			step = get_step(a, b, x, y, interval, delta)
			x = x + step*delta[0]
			y = y + step*delta[1]
			print("point: " + str(x) + ', ' + str(y))
			delta = [0, 0]
			pre[2] = now[2]
			now[2] = Rosenbrock(a, b, x, y)
			print("function value: " + str(now[2]))
		# 考察每个方向的变化
		if math.fabs(pre[0]-now[0]) <= eps and\
			math.fabs(pre[1]-now[1]) <= eps and\
			math.fabs(pre[2]-now[2]) <= eps:
			break
		n = n + 1

	min_point = [x, y]
	print("total times: " + str(n))

	return min_point

if __name__ == '__main__':

	print(cyclic(1, 100, [10,10], [-3,8], 0.0001))