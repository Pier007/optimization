# -*- coding: utf-8 -*-

import math
import numpy as np

# rosenbrock funtion
def Rosenbrock(a, b, x, y):

	return (a-x)*(a-x) + b*(y-x*x)*(y-x*x)

# 泰勒二次展开
def Rosenbrock_q(a, b, x, y, delta_x, delta_y, grad, hessian):

	delta = np.array([delta_x, delta_y])

	val = Rosenbrock(a, b, x, y)
	val += np.dot(grad, delta)
	val += 0.5*np.dot(np.dot(delta, hessian), delta)

	return val

# objective function
def func(params, p, lam):

	a = params[0]
	b = params[1]
	x = params[2]
	y = params[3]

	x = x + lam*p[0]
	y = y + lam*p[1]

	return Rosenbrock(a, b, x, y)

# 黄金分割法
def goldenSelection(interval, eps, func, params, p):

	t = (math.sqrt(5)-1)/2
	a = interval[0]
	b = interval[1]

	r = a + (1-t)*(b-a)
	u = a + t*(b-a)

	if p == [0, 0]:
		return 0

	while u-r > eps:
		if func(params, p, r) > func(params, p, u):
			a = r
			r = u
			u = a + t*(b-a)
		else:
			b = u
			u = r
			r = a + (1-t)*(b-a)
	
	x = (u+r)/2

	return x

# 计算梯度
def get_grad(a, b, x, y):

	grad = [2*(x-a)+4*b*(x*x-y)*x,
		2*b*(y-x*x)]
	e = math.sqrt(math.pow(grad[0],2) + math.pow(grad[1],2))

	for i in range(len(grad)):
		if e == 0:
			grad[i] = 0
		else:
			# 归一化
			grad[i] = grad[i] / e
	return grad

# 计算海塞矩阵
def get_hessian(b, x, y):

	hessian = [[0, 0], [0, 0]]

	hessian[0][0] = 2+4*b*(x**2-y)+8*b*x**2
	hessian[0][1] = -4*b*x
	hessian[1][0] = -4*b*x
	hessian[1][1] = 2*b

	return hessian

# 计算最优步长
def get_step(a, b, x, y, interval, p):

	eps = 0.01
	params = [a, b, x, y]

	step = goldenSelection(interval, eps, func, params, p)

	return step