'''
**************************************蒙特卡洛模拟部分*******************************************
1.产生均匀随机数的方法：平方取中法、乘同余法
2.产生具有给定分布的随机变量：随机抽样
  2.1连续型分布直接抽样
  2.2离散型分布直接抽样
  2.3变换抽样法
  2.4舍选法
  2.5截尾分布
3.蒙特卡洛模拟炮弹落入椭圆区域内概率
4.蒙特卡洛模拟定积分数值解计算
5.蒙特卡洛用于求解整数规划
6.蒙特卡洛用于求解偏微分方程的数值解



'''
from time import time
import numpy as np
from numpy import sqrt
from numpy.random import uniform
from scipy.integrate import dblquad,quad
import matplotlib.pyplot as plt
def rander(seed, n):
    '''平方取中法'''
    if n == 1:
        return 0
    seed = int(seed)
    length = len(str(seed))
    seed = int(seed ** 2 / pow(10, (length / 2))) % int(pow(10.0, length))
    print(str(seed) + " ", end='')
    rander(seed, n - 1)
def mengte(x):
    f = sum(x ** 2) + np.array([-8, -2, -3, -1, -2]).dot(x)
    g = np.array([[sum(x) - 400],
                  [np.array([1, 2, 2, 1, 6]).dot(x) - 800],
                  [np.array([2, 1, 6, 0, 0]).dot(x) - 200],
                  [np.array([0, 0, 1, 1, 5]).dot(x) - 200]])
    return (f, g)

if __name__ == '__main__':
    print('*'*50+'蒙特卡洛模拟'+'*'*50)

    #平方取中法
    print('-'*50+'1.平方取中法'+'-'*50)
    rander(time(),10)

    print('-'*50+'3.炮弹落入椭圆区域内概率模拟'+'-'*50)
    mu=[0,0]; cov=10000*np.array([[1,0.5],[0.5,1]]);
    N=1000000
    fxy=lambda y,x: 1/(20000*np.pi*np.sqrt(0.75))*\
         np.exp(-1/1.5*(x**2-x*y+y**2)/10000)  #接上一行
    bdy=lambda x: 80*np.sqrt(1-x**2/120**2)
    p1=dblquad(fxy,-120,120,lambda x:-bdy(x),bdy)[0]
    print("概率的数值解为：",p1)
    a=np.random.multivariate_normal(mu,cov,size=N)
    n=((a[:,0]**2/120**2+a[:,1]**2/80**2)<=1).sum()
    p2=n/N; print('概率的近似值为：',p2)


    print('-'*50+'4.计算定积分的近似解'+'-'*50)
    #4.1 积分示例一样本平均法：
    print('+'*30+'示例一'+'+'*30)
    y = lambda x: x / np.sqrt(5 - 4 * x)  # 定义被积函数的匿名函数
    I = quad(y, -1, 1)[0]  # 计算积分的数值解，与随机模拟得到的解进行对比
    n = 100000000  # 生成随机数的个数
    x = np.random.uniform(-1, 1, size=n)  # 生成区间（-1,1）上均匀分布的n个随机数
    h = y(x)  # 计算被积函数的一系列取值
    junzhi = h.mean()  # 计算取值的平均值
    jifen = 2 * junzhi  # 计算积分的近似值
    print("数值解为：", I);
    print("样本平均法模拟的数值解为：", jifen)

    #积分示例一随机投针法
    yx = lambda x: x / sqrt(5 - 4 * x)  # 定义被积函数的匿名函数
    n = 10 ** 6  # 生成随机数的个数
    x0 = uniform(-1, 1, size=n)  # 生成n个区间[-1,1)上的随机数
    y0 = uniform(-1, 1, size=n)  # 这里直接产生[-1,1)上的随机数，不做变换
    n0 = sum(y0 < yx(x0));
    I = n0 / n * 4 - 2
    print("随机投针法模拟的数值解为：", I)


    #4.2 积分示例二样本平均法
    print('+'*30+'示例二'+'+'*30)
    yx = lambda x: np.cos(x) * np.exp(-x ** 2 / 2)
    I1 = quad(yx, -np.inf, np.inf)[0]  # 其数值积分
    r = np.random.normal(size=10 ** 7)  # 生成标准正态分布随机数
    I2 = np.sqrt(2 * np.pi) * np.cos(r).mean()
    print("数值解为：", I1);
    print("样本平均法模拟的数值解为：", I2)


    #5.求解整数规划：效果没有Lingo的全局最优解好
    print('-'*30+'5.蒙特卡洛用于求解整数规划'+'-'*30)
    p0 = 0
    for i in range(10 ** 6):
        x = np.random.randint(0, 100, 5)
        f, g = mengte(x)
        if all(g <= 0):
            x0 = x;
            p0 = f
    print("近似最优解：", x0)
    print("近似最优解：", p0)
    #近似最优解： [22 24 10 60  1]
    #近似最优解： 4445

    #求解偏微分方程数值解
    print('-'*30+'6.蒙特卡洛用于求解偏微分方程的数值解'+'-'*30)
    x=np.linspace(0,1,101); y=np.linspace(0,1,101)
    phi=np.zeros(400)  #边界条件初始化
    phi[:101]=10; N=1000
    u=np.zeros((101,101)); u[:,0]=10  #初始化
    #内部节点的编号i=1,2,…，99；j=1,2,…，99
    s=np.zeros((101,101))
    for i in range(1,99):
        for j in range(1,99):
            for k in range(N):
                s[i,j]=0; ii=i; jj=j
                while (ii>0) & (ii<100) & (jj>0) & (jj<100):
                    r=np.random.randint(1,5)  #生成1,2,3,4中的一个随机整数
                    ii=ii+(r==1)-(r==3)
                    jj=jj+(r==2)+(r==4)
                if jj==100: kk=ii
                elif ii==100: kk=100+(100-jj)
                elif jj==0: kk=200+(100-ii)
                else: kk=300+jj
                s[i,j]=s[i,j]+phi[kk]
            u[i,j]=s[i,j]/N
    plt.contour(x,y,u); plt.show()





