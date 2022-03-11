'''
**************************************Sympy基础*******************************************
'''
from IPython.display import display
import sympy as sp
import sympy
from sympy import *
import matplotlib.pyplot as plt
from scipy.integrate import quad

def toeplitz(n):
    '''托普利兹矩阵:主对角线上的元素相等，平行于主对角线的线上的元素也相等；
    矩阵中的各元素关于次对角线对称，即T型矩阵为次对称矩阵。'''
    a = sp.symbols('a:' + str(2 * n - 1))
    f = lambda i, j: a[i - j + n - 1]
    return sp.Matrix(n, n, f)

def conditions(args):
    '''优化化问题中的限制条件'''
    x1min, x1max, x2min, x2max, x3min, x3max = args
    cons = ({'type': 'ineq', 'fun': lambda x: x[0] - x1min},
            {'type': 'ineq', 'fun': lambda x: -x[0] + x1max},
            {'type': 'ineq', 'fun': lambda x: x[1] - x2min},
            {'type': 'ineq', 'fun': lambda x: -x[1] + x2max},
            {'type': 'ineq', 'fun': lambda x: x[2] - x3min},
            {'type': 'ineq', 'fun': lambda x: -x[2] + x3max})
    return cons


if __name__ == '__main__':
    print('#'*30+'Sympy基础'+'#'*30)

    '''1.Sympy
    1.1. 符号变量的定义：符号变量、符号函数、符号矩阵、字符串确定表达式
    1.2. Lambda function、符号替换
    1.3. 公式推导：多项式展开、折叠、因式分解、合并同类项、求因式、化简、有理分式化简
    1.4. 符号替换、符号的值转化、公式简化
    1.5. 符号函数画图：二维、三维、隐函数绘图、参数方程绘图
    1.6. 求解方程：普通代数方程、线性方程组、常微分方程、偏微分方程、优化问题
    1.7. 数学计算:求和、求导(偏导)、积分(不定、定、多重不定、多重定)
    1.8. 数学契合：一些数学常量表示
    '''
    #1.1.符号变量的定义
    print('*'*50+'符号变量的定义'+'*'*50)
    print('\033[1;45m 这部分自己调试学习写法 \033[0m')
    x,y,z=symbols('x y z')      #等效于x,y,z=symbols('x,y,z'),Symbol对象
    #整组符号的定义
    integer_variables=symbols('I:L',interger=True)  #tuple(Symbol)
    real_variables=symbols('a:d',real=True)         #tuple(Symbol)
    A=symbols('A1:3(1:4)')                          #tuple(Symbol)
    #符号函数
    f=Function('f')
    f,g=symbols('f g',cls=Function) #同时定义多个符号函数
    #符号矩阵
    print('符号矩阵：')
    M = sp.Matrix(3, 3, sp.symbols('m:3(:3)'))
    print(M)
    T = toeplitz(5);
    print(T)
    #定义多元函数示例
    print('定义多元函数：')
    x = symbols('x:3')  # 输出：(x0, x1, x2)
    f = Function('f')  # 定义符号函数
    print(f(*x))  # 输出：f(x0, x1, x2)
    #根据字符串确定表达式
    expr='x**2+2*x+1'
    expr=simplify(expr)
    print(expr)

    #1.2. Lambda function
    print('*'*50+'Lambda函数'+'*'*50)
    #Lambda表达式的创建与带入值计算
    x,y,c,rho,a,v=sp.symbols('x,y,c,rho,a,v')
    f=sp.Lambda(v,-sp.Rational(1,2)*c*rho*a*v**2)
    z=sp.Lambda((x,y),sp.sin(x)+sp.cos(2*y))
    print(f'这是关于{f.args[0]}的方程:{f.args[1]}')    #f.args[0]表示自变量元组，可能有一个或多个；f.args[1]表示公式，是公式对象只有一个式子
    v=1
    print(f'带入数值v={v}计算结果={f(v)}')
    x=1
    y=2
    print(f'这是关于{z.args[0]}的方程:{z.args[1]}')
    print(f'带入数值(x,y)={(x,y)}计算结果={z(x,y)}')


    #Lambda表达式可以通过rcall函数对自变量进行整体替换，例如:f=x^2,f.rcall(xsin(x))=x**2*sin(x)**2
    from sympy.abc import x, y, z   #重要，不能少
    sympy.init_printing(use_latex='mathjax', forecolor='White')
    sympy.init_printing(use_latex=True, forecolor='White')

    f0 = Lambda(y, 2 * sin(y ** 2))
    f1 = Lambda(y, sin(x) + sin(2 * x))
    f = x + f0 + f1
    print('原函数:', f)
    result1 = f.rcall(sin(x) * x)
    print('使用xsinx替换x:', result1)
    result2 = f.rcall(sin(x) * x).subs({x: 1}).evalf()
    print('使用xsinx替换x并代入x=1:', result2)

    # 可以通过星号表达式的方法把最后表达式输出来，就不用一个一个写参数了
    p = x, y, z
    f = Lambda(p, x + y * z)
    print('原输出:',f)
    print('使用星号表达式输出:',f(*p))


    #雅可比行列式
    print('*'*50+'雅可比行列式'+'*'*50)
    #方式一：正常定义x与y
    x,y=sp.symbols('x,y')
    F=sp.Lambda((x,y),sp.Matrix([sp.sin(x)+sp.cos(2*y),sp.sin(x)*sp.cos(y)]))
    J=F(x,y).jacobian((x,y))
    print(J)

    #方式二：简便定义自变量
    x = sp.symbols('x:2')
    F = sp.Lambda(x, sp.Matrix([sp.sin(x[0]) + sp.cos(2 * x[1]), sp.sin(x[0]) * sp.cos(x[1])]))
    J = F(*x).jacobian(x)
    print(J)

    # 1.3.公式推导
    print('*'*50+'公式推导'+'*'*50)
    # 展开
    x, y, z = symbols('x y z')
    y = expand((x + y + 1) ** 2+(x + y + 1))
    print('展开：',y)
    # 折叠
    y=y.factor()
    print('折叠：',y)
    # 因式分解
    z = factor(y)
    print('因式分解:',z)
    # 合并同类项
    g = collect(y, x)
    print('合并同类项：',g)
    # 因式拆解
    p = apart(1 / ((1 + x) * (3 + x)))
    print('因式拆解：',p)
    # 化简
    s = simplify(2 * sin(x) * cos(x))
    print('化简：', p)
    # 有理分式化简
    s = cancel((x ** 2 + 2 * x + 1) / (x ** 2 + x))
    print('有理分式化简：', p)

    #1.4.符号替换、符号的值转化、公式简化
    '''
    a.subs(x,0)
    a.subs({x:0,a:1})       多替换
    a.subs([(x,0),(a,1)])   （旧值,新值）对照表
    '''
    print('*'*50+'符号替换'+'*'*50)
    x,a=sp.symbols('x,a')
    b=x+a;
    c=b.subs(x,0)
    d=c.subs(a,2*a)
    print(c,d)

    #1.4.1 利用T矩阵构造三对角线矩阵
    T = toeplitz(5)
    symbs = [sp.symbols('a' + str(i)) for i in range(9) if i < 3 or i > 5]
    substitution = list(zip(symbs, [0] * len(symbs)))
    T0 = T.subs(substitution);
    print(T0)

    #1.4.2符号值转化成浮点型：使用.evalf()或者.n()方法
    x1=sp.sin(1)
    x2=x1.evalf()  #转换为浮点值
    x3=x1.n()       #x2与x3等效
    print(x1)
    print(x2)  #显示符号值和浮点值
    print(x3)  #显示符号值和浮点值

    # 运用evalf函数传值
    x = Symbol('x')
    fx = 5 * x + 4
    y1 = fx.evalf(subs={x: 6})
    print(y1)

    # 多元表达式运用evalf函数传值
    x = Symbol('x')
    y = Symbol('y')
    fx = x * x + y * y
    result = fx.evalf(subs={x: 3, y: 4})
    print(result)

    #1.4.3 公式简化
    # simplify( )一般的化简
    normal_simplify =simplify((x ** 3 + x ** 2 - x - 1) / (x ** 2 + 2 * x + 1))
    print('普通简化:',normal_simplify)

    # trigsimp( )三角化简
    trig_simplify=trigsimp(sin(x) / cos(x))
    print('三角简化:',trig_simplify)

    # powsimp( )指数化简
    pow_simplify=powsimp(x ** a * x ** b)
    print('指数简化:',pow_simplify)

    #1.5. 符号函数画图
    print('*'*50+'符号函数画图'+'*'*50)
    #1.5.1 二维曲线图：在同一个图上画出
    # y1=2sin(2x),-3<=x<=3
    # y2=cos(x+pi/4),-4<=x<=4
    x,pi=sp.symbols('x,pi');
    sp.plot((2*sp.sin(2*x),(x,-3,3)),(sp.cos(x+pi/3),(x,-4,4)),
            xlabel='$x$',ylabel='$y$')


    #1.5.2 三维曲面画图
    # 画出z=cos(2sqrt(x^2+y^2)+ln(x^2+y^2+1))
    from sympy.plotting import plot3d
    plt.rc('font',size=16)
    plt.rc('text',usetex=True)
    x,y=sp.symbols('x,y');
    plot3d(sp.cos(2*sp.sqrt(x**2+y**2))+sp.log(x**2+y**2+1),
           (x,-10,10),(y,-10,10),xlabel='$x$',ylabel='$y$')

    #1.5.3 隐函数画图
    #方法一：from sympy import plot_implicit as pt,Eq
    from sympy import plot_implicit as pt
    from sympy.abc import x,y   #引进符号变量x,y
    pt(Eq((x-1)**2+(y-2)**3,4),(x,-6,6),(y,-2,4))

    #方法二：使用匿名函数Lambda构造
    ezplot = lambda expr: pt(expr)
    ezplot((x - 1) ** 2 + (y - 2) ** 3 - 4)

    #1.5.4 参数方程绘图
    from sympy.plotting import plot_parametric,plot3d_parametric_line,plot3d_parametric_surface
    u = symbols('u')
    plot_parametric(cos(u), sin(u), (u, -5, 5))
    plot3d_parametric_line(cos(u), sin(u), u, (u, -5, 5))

    u, v = symbols('u v')
    plot3d_parametric_surface(cos(u + v), sin(u - v), u - v, (u, -5, 5), (v, -5, 5))

    #1.6.求解方程
    print('*'*50+'求解方程'+'*'*50)

    # 1.6.1 求解普通方程（无论是有数值解还是没有数值解都能解）
    x = symbols('x')
    result = solve('x**2 - 2*x + 1', x)
    print('普通方程有数值解：')
    print(result)
    # 这个是没有数值解的情况
    x, y = symbols('x,y')
    result = solve('x**2 - 2*x + 1 + y', x, y)
    print('普通方程没有数值解：')
    print(result)

    # 1.6.2 求解线性方程组
    import numpy as np
    from scipy.linalg import solve  # 注意：矩阵相关的方程就要使用scipy.linalg.solve，普通求解使用sp.solve即可

    A = np.array([[3, 1, -2], [1, -1, 4], [2, 0, 3]])
    B = np.array([10, -2, 9])
    result = solve(A, B)
    print('求解线性方程组：')
    print(result)

    # 1.6.3 求解高阶常微分方程
    from sympy.interactive import printing

    printing.init_printing(use_latex=True)
    print('常微分方程求解：')

    x = symbols('x', real=True)  # real 保证全是实数，自变量
    y = symbols('y', function=True, cls=Function)  # 全部为函数变量
    eq = y(x).diff(x, 4) - 2 * y(x).diff(x, 3) + 5 * y(x).diff(x, 2)
    display(eq)  # 这个用于jupyter里面
    result = dsolve(Eq(eq, 0), y(x))
    print(f'{result.args[0]}={result.args[1]}')  # Eq(eq,0)等效为等式的左右两边，dsolve(公式，求解对象)


    #1.6.4 偏微分方程
    '''题目在这里：https://i.stack.imgur.com/AlboM.png'''
    x1, x2 = sp.symbols('x1, x2')
    f = sp.Function('f')
    phi = f(x1, x2)
    eq = phi.diff(x1) * sp.cos(x2) + phi.diff(x2)
    print('偏微分方程的解为:',sp.pdsolve(eq))  # f(x1, x2) == F(-x1 + sin(x2))

    # 1.6.5 优化问题
    # 例题：min (2+x1)/(1+x2)-3x1+4x3
    # 0.1<=x1<=0.9
    # 0.1<=x2<=0.9
    # 0.1<=x3<=0.9
    from scipy.optimize import minimize
    import numpy as np

    f = lambda x: (2 + x[0]) / (1 + x[1]) - 3 * x[0] + 4 * x[2]
    args1 = (0.1, 0.9, 0.1, 0.9, 0.1, 0.9)  # x1min, x1max, x2min, x2max
    cons = conditions(args1)

    x0 = np.array([0.5, 0.5, 0.5])
    result = minimize(f, x0, method='SLSQP', constraints=cons)
    print('优化问题：')
    print(result)

    # 1.7.数学计算
    print('*' * 50 + '数学计算' + '*' * 50)

    # 1.7.1 求和
    n = Symbol('n')
    f = 2 * n
    s = summation(f, (n, 1, 100))  # summation(函数, (自变量, 下限, 上限))
    print('求和结果:', s)



    # # 解求和方程（程序没问题，单独开一个.py运行）
    # x = Symbol('x')
    # i = Symbol('i')
    # f = summation(x, (i, 1, 5)) + 10 * x - 15  # 解释一下，i能够看做是循环变量，便是x自己加五次
    # result = solve(f, x)
    # print('求和方程结果:', result)



    # 1.7.2 求极限
    # 求极限运用limit办法
    x = Symbol('x')
    f1 = sin(x) / x
    f2 = (1 + x) ** (1 / x)
    f3 = (1 + 1 / x) ** x
    lim1 = limit(f1, x, 0)  # 三个参数是 函数，变量，趋向值
    lim2 = limit(f2, x, 0)
    lim3 = limit(f3, x, oo)  # infsympy.oo表示
    print('求极限结果:', lim1, lim2, lim3)

    # 1.7.2 求导
    # 求导运用diff办法
    x = Symbol('x')
    f1 = 2 * x ** 4 + 3 * x + 6
    # 参数是函数与变量
    f1_ = diff(f1, x)
    print('求导结果：')
    print(f1_)

    f2 = sin(x)
    f2_ = diff(f2, x)
    print(f2_)

    # 求偏导
    y = Symbol('y')
    f3 = 2 * x ** 2 + 3 * y ** 4 + 2 * y
    # 对x，y别离求导，即偏导
    f3_x = diff(f3, x)
    f3_y = diff(f3, y)
    print('diff求偏导：')
    print(f3_x)
    print(f3_y)

    x, y, z = symbols('x y z')
    z = 2 * x ** 2 + 3 * y ** 4 + 2 * x * y
    # 对x，y分别求导，即偏导
    z_x = Derivative(z, x, x)  # Derivative(函数, 第一次求导, 第二次求导,...)
    z_y = Derivative(z, y, x)
    print('Derivative求偏导：')
    print('z_x=', z_x.doit())
    print('z_y=', z_y.doit())

    # 用subs()函数求指定位置的导数/偏导值
    x, y = symbols('x y', real=True)
    print(diff(x ** 2 + y ** 3, y).subs({x: 3, y: 1}))  # 3

    # 1.7.3 积分

    # 不定积分
    # 求不定积分其实和定积分差异不大
    x = Symbol('x')
    f = (E ** x + 2 * x)
    f_ = integrate(f, x)
    print('不定积分结果:', f_)

    # 定积分
    import scipy

    x = symbols('x')
    f = Lambda(x, x + 1)
    v, err = quad(f, 1, 2)  # err为差错
    print('一重定积分结果:', v)

    # 多重定积分
    '''积分这块可以参考使用scipy.integrate:
    https://blog.csdn.net/t4ngw/article/details/105739379
    sympy积分可以参考：https://www.jishulink.com/content/post/1188088
    '''
    x, t = symbols('x t')
    f1 = 2 * t
    f2 = sp.integrate(f1, (t, 0, x))
    result = sp.integrate(f2, (x, 0, 3))
    print('多重定积分结果:', result)

    # 多重不定积分
    p = Derivative(x ** 5, x, 3)
    q = integrate(p, x, x, x)
    print('多重不定积分结果：', q)



    # 1.8.数学契合
    print('*' * 50 + '数学契合' + '*' * 50)
    # 虚数单位i
    sympy.I
    # 自然对数低e
    sympy.E
    # 无量大
    sympy.oo
    # 圆周率
    sympy.pi
    # 求n次方根
    sympy.root(8, 3)
    # 求对数
    sympy.log(1024, 2)
    # 求阶乘
    sympy.factorial(4)
    # 三角函数
    sympy.sin(sympy.pi)
    sympy.tan(sympy.pi / 4)
    sympy.cos(sympy.pi / 2)







