# from IPython.display import display
from scipy.integrate import quad
from sympy import *
import sympy as sp
# # 1.5.1 求解普通方程（无论是有数值解还是没有数值解都能解）
# x = symbols('x')
# result = solve('x**2 - 2*x + 1', x)
# print('普通方程有数值解：')
# print(result)
# # 这个是没有数值解的情况
# x, y = symbols('x,y')
# result = solve('x**2 - 2*x + 1 + y', x, y)
# print('普通方程没有数值解：')
# print(result)
#
# #1.5.2 求解线性方程组
# import numpy as np
# from scipy.linalg import solve    #注意：矩阵相关的方程就要使用scipy.linalg.solve，普通求解使用sp.solve即可
# A = np.array([[3, 1, -2], [1, -1, 4], [2, 0, 3]])
# B = np.array([10, -2, 9])
# result = solve(A, B)
# print('求解线性方程组：')
# print(result)
#
# # 1.5.3 求解高阶常微分方程
# from sympy.interactive import printing
# printing.init_printing(use_latex=True)
# print('常微分方程求解：')
#
# x=symbols('x',real = True) # real 保证全是实数，自变量
# y=symbols('y',function = True,cls=Function) # 全部为函数变量
# eq=y(x).diff(x,4)-2*y(x).diff(x,3)+5*y(x).diff(x,2)
# display(eq) #这个用于jupyter里面
# result=dsolve(Eq(eq,0),y(x))
# print(f'{result.args[0]}={result.args[1]}')    #Eq(eq,0)等效为等式的左右两边，dsolve(公式，求解对象)
#
# #1.5.4 优化问题
# # 例题：min (2+x1)/(1+x2)-3x1+4x3
# # 0.1<=x1<=0.9
# # 0.1<=x2<=0.9
# # 0.1<=x3<=0.9
# from scipy.optimize import minimize
# import numpy as np
#
# f = lambda x : (2 + x[0]) / (1 + x[1]) - 3*x[0] + 4*x[2]
#
# def con(args):
#     x1min, x1max, x2min, x2max, x3min, x3max = args
#     cons = ({'type': 'ineq', 'fun': lambda x: x[0] - x1min},
#             {'type': 'ineq', 'fun': lambda x: -x[0] + x1max},
#             {'type': 'ineq', 'fun': lambda x: x[1] - x2min},
#             {'type': 'ineq', 'fun': lambda x: -x[1] + x2max},
#             {'type': 'ineq', 'fun': lambda x: x[2] - x3min},
#             {'type': 'ineq', 'fun': lambda x: -x[2] + x3max})
#     return cons
#
# args1 = (0.1, 0.9, 0.1, 0.9, 0.1, 0.9)  # x1min, x1max, x2min, x2max
# cons = con(args1)
#
# x0 = np.array([0.5, 0.5, 0.5])
# result = minimize(f, x0, method='SLSQP', constraints=cons)
# print('优化问题：')
# print(result)
#

#
# # 1.7.数学计算
# print('*' * 50 + '数学计算' + '*' * 50)
#
# # 1.7.1 求和
# n = Symbol('n')
# f = 2 * n
# s = summation(f, (n, 1, 100))  # summation(函数, (自变量, 下限, 上限))
# print('求和结果:', s)
#
# # 解求和方程
# x = Symbol('x')
# i = Symbol('i')
# f = summation(x, (i, 1, 5)) + 10 * x - 15  # 解释一下，i能够看做是循环变量，便是x自己加五次
# result = solve(f, x)
# print('求和方程结果:', result)
#
# # 1.7.2 求极限
# # 求极限运用limit办法
# x = Symbol('x')
# f1 = sin(x) / x
# f2 = (1 + x) ** (1 / x)
# f3 = (1 + 1 / x) ** x
# lim1 = limit(f1, x, 0)  #三个参数是 函数，变量，趋向值
# lim2 = limit(f2, x, 0)
# lim3 = limit(f3, x, oo) #infsympy.oo表示
# print('求极限结果:', lim1, lim2, lim3)
#
# # 1.7.2 求导
# # 求导运用diff办法
# x = Symbol('x')
# f1 = 2 * x ** 4 + 3 * x + 6
# # 参数是函数与变量
# f1_ = diff(f1, x)
# print('求导结果：')
# print(f1_)
#
# f2 = sin(x)
# f2_ = diff(f2, x)
# print(f2_)
#
# # 求偏导
# y = Symbol('y')
# f3 = 2 * x ** 2 + 3 * y ** 4 + 2 * y
# # 对x，y别离求导，即偏导
# f3_x = diff(f3, x)
# f3_y = diff(f3, y)
# print('diff求偏导：')
# print(f3_x)
# print(f3_y)
#
# x, y, z = symbols('x y z')
# z = 2 * x ** 2 + 3 * y ** 4 + 2 * x * y
# # 对x，y分别求导，即偏导
# z_x = Derivative(z, x, x)   #Derivative(函数, 第一次求导, 第二次求导,...)
# z_y = Derivative(z, y, x)
# print('Derivative求偏导：')
# print('z_x=',z_x.doit())
# print('z_y=',z_y.doit())
#
# # 用subs()函数求指定位置的导数/偏导值
# x, y = symbols('x y', real=True)
# print(diff(x ** 2 + y ** 3, y).subs({x: 3, y: 1}))  # 3
#
# #1.7.3 积分
#
# #不定积分
# #求不定积分其实和定积分差异不大
# x=Symbol('x')
# f=(E**x+2*x)
# f_=integrate(f,x)
# print('不定积分结果:',f_)
#
# #定积分
# import scipy
# x=symbols('x')
# f=Lambda(x,x+1)
# v, err = quad(f, 1, 2)# err为差错
# print('一重定积分结果:',v)
#
# #多重定积分
# '''积分这块可以参考使用scipy.integrate:
# https://blog.csdn.net/t4ngw/article/details/105739379
# sympy积分可以参考：https://www.jishulink.com/content/post/1188088
# '''
# x,t=symbols('x t')
# f1=2*t
# f2=sp.integrate(f1,(t,0,x))
# result=sp.integrate(f2,(x,0,3))
# print('多重定积分结果:',result)
#
#
# #多重不定积分
# p = Derivative(x**5, x, 3)
# q = integrate(p, x, x, x)
# print('多重不定积分结果：',q)

from sympy import Lambda
from sympy import *
from sympy.abc import x, y,z
import sympy

sympy.init_printing(use_latex='mathjax', forecolor='White')
sympy.init_printing(use_latex=True, forecolor='White')
sympy.init_printing()


# f0 = Lambda(y, 2*sin(y**2))
# f1 = Lambda(y, sin(x)+sin(2*x))
# f = x + f0 + f1
# print('原函数:',f)
# result1=f.rcall(sin(x)*x)
# print('使用xsinx替换x:',result1)
#
# result2=f.rcall(sin(x)*x).subs({x:1}).evalf()
# print('使用xsinx替换x并代入x=1:',result2)
#
# #
# f0 = Lambda((x, y),x+y)
#
# #
# f = Lambda(x, x**2)
# f,f(0)
#
# #
# x,y,z,t = symbols("x,y,z,t")
# f2 = Lambda((x, y, z, t), x + y**z + t**z)
# f2,f2(1, 2, 3, 4)
#
# #
# f = Lambda( ((x, y), z) , x + y + z)
# f,f((1, 2), 3)
#
# #可以通过星号表达式的方法把最后表达式输出来，就不用一个一个写参数了
# p = x, y, z
# f = Lambda(p, x + y*z)
# print(f)
# print(f(*p))
#

# x1, x2 = sp.symbols('x1, x2')
# f = sp.Function('f')
# phi = f(x1, x2)
# eq = phi.diff(x1) * sp.cos(x2) + phi.diff(x2)
# print(sp.pdsolve(eq)) # f(x1, x2) == F(-x1 + sin(x2))



n = 5000
i = Symbol('i',dtype=Integer)
f=i*sqrt(pow(n,4)-pow(i,4))/pow(n,4)
f_sum = summation(f, (i, 1, n))  # 解释一下，i能够看做是循环变量，便是x自己加五次
print(f_sum)


