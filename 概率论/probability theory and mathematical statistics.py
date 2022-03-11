'''
**************************************概率论部分*******************************************
概念速查：
* PDF：概率密度函数（probability density function）
在数学中，连续型随机变量的概率密度函数（在不至于混淆时可以简称为密度函数）是一个描述这个随机变量的输出值，在某个确定的取值点附近的可能性的函数。
* PMF : 概率质量函数（probability mass function)
在概率论中，概率质量函数是离散随机变量在各特定取值上的概率。
* CDF : 累积分布函数 (cumulative distribution function)
又叫分布函数,是概率密度函数的积分，能完整描述一个实随机变量X的概率分布。
'''

import math
from scipy.special import *
from numpy.random import uniform
import pylab as plt
import random
from scipy.stats import gamma
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate  # 插值平滑
from sklearn.linear_model import LinearRegression
from statsmodels.distributions import ECDF
import scipy.stats as st


def draw_gamma(x,alpha,beta):
    plt.rc('text', usetex=True);
    plt.rc('font', size=15)
    s = ['*-', 'o-', '>-']
    plt.plot(x, gamma.pdf(x, alpha, beta), s[random.randrange(3)])
    plt.legend([f'$\\alpha={alpha},\\beta={beta}$'])
    plt.xlabel('$x$');
    plt.ylabel('$f(x)$')
    plt.show()

def getPoisson(lam,size):
    '''返回服从参数为lam的泊松分布的一组整数np数组，元素个数为size'''
    int_seq=np.random.poisson(lam,size)
    #统计不同元素的个数
    freq={}
    for i in int_seq:
        if(i not in freq): freq[i]=0
        freq[i]+=1
    arr=list(freq.keys())
    arr.sort() #将数字升序排列
    x_data=np.array(arr) #x坐标点
    y_data=np.array([freq[i] for i in x_data]) #y坐标
    return x_data,y_data

def showPoisson(data,itrplt_type='cubic'):
    '''显示一组泊松分布图，默认平滑插值类型是'cubic'，如果为None则不平滑'''
    plt.figure()
    for lam,size,color in data:
        x,y=getPoisson(lam,size)    #lam就是lambda：区域中事件发生的均值；size就是区域内发生特定事件的次数
        if(itrplt_type):
            func=interpolate.interp1d(x,y,kind=itrplt_type)
            xnew=np.arange(np.min(x),np.max(x),0.01)
            ynew=func(xnew)
            plt.plot(xnew,ynew,color=color,linewidth=2)
        else:
            plt.plot(x,y,color=color,linewidth=2)
    plt.show()

if __name__ == '__main__':
    print('*'*50+'概率论部分'+'*'*50)

    '''随机事件机器概率
    math.factorial(n)        #计算阶乘n!
    math.gamma(n+1)          #计算阶乘n!
    scipy.special.comb(n,k)  #计算C_n^k
    '''

    #1.蒲丰投针实验
    a = 45;
    L = 36;
    n = 100000
    x = uniform(0, a / 2, n);  # 产生n个[0,a/2)区间上均匀分布的随机数
    phi = uniform(0, np.pi, n)
    m = sum(x <= L * np.sin(phi) / 2)
    pis = 2 * n * L / (a * m)  # 计算pi的近似值
    print('π的近似值为：',pis)
    print('*'*100)

    '''统计模块scipy.stats
    scipy.stats官网:      https://docs.scipy.org/doc/scipy/reference/stats.html
    Gamma分布和Beta分布代码:https://gist.github.com/wisimer/87936ebbf3619956f6b9c8557e106c38#file-betadistribution-ipynb
    
    @可能用到的分布对照表:
    beta	    beta分布
    f	        F分布
    gamma	    gam分布
    poisson	    泊松分布
    hypergeom	超几何分布
    lognorm	    对数正态分布
    binom	    二项分布
    uniform	    均匀分布
    chi2	    卡方分布
    cauchy	    柯西分布
    laplace	    拉普拉斯分布
    rayleigh	瑞利分布
    t	        学生T分布
    norm	    正态分布
    expon	    指数分布
    
    @stats连续型随机变量的公共方法：
    rvs	        产生服从指定分布的随机数
    pdf	        概率密度函数
    cdf	        累计分布函数
    sf	        残存函数（1-CDF）
    ppf	        分位点函数（CDF的逆）
    isf	        逆残存函数（sf的逆）
    fit	        对一组随机取样进行拟合，最大似然估计方法找出最适合取样数据的概率密度函数系数。
    '''
    #2.绘制gamma分布的概率密度曲线图
    x = np.linspace(0, 20, 100)
    draw_gamma(x,1,3)

    '''
    泊松分布计算与绘图:https://blog.csdn.net/lanhezhong/article/details/105765526
    通俗理解泊松分布: https://blog.csdn.net/ccnt_2012/article/details/81114920
    '''


    ##3.1绘制泊松分布分布律图形(由于是离散的，使用cubic插值)
    showPoisson(data=[(5,10000,'#990000'),(5,20000,'#D64700'),(5,40000,'#006699')],itrplt_type='cubic')
    ##输入以列表形式，第一个参数是λ：区域中事件发生的均值；第二个参数是X：区域内发生特定事件的次数




    #3.2比较泊松分布与正态分布：当λ≥20时，泊松分布可以用正态分布来近似，当λ≥50，泊松分布基本上就等于正态分布了。
    # 此时μ=σ^2=λ
    # 由此可见，当离散数据的值足够大时，可以当成连续数据来分析。
    from numpy import random
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    sns.distplot(random.normal(loc=50, scale=7, size=1000), hist=False, label='normal')
    sns.distplot(random.poisson(lam=50, size=1000), hist=False, label='poisson')
    plt.legend(['正态分布', '泊松分布'])
    # plt.show()


    #3.3比较泊松分布与二项分布：用泊松分布来近似计算可以降低大量的计算量。
    # 近似时，λ=np，一般来讲，当n很大p很小时(n≥100，np≤10)近似效果较好
    from numpy import random
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['SimHei']
    sns.distplot(random.binomial(n=1000, p=0.01, size=1000), hist=False, label='binomial')
    sns.distplot(random.poisson(lam=10, size=1000), hist=False, label='poisson')
    plt.legend(['二项分布', '泊松分布'])
    # plt.show()


    #4.一维随机变量的计算
    #4.1 计算正态分布累积概率密度函数cdf
    from scipy.stats import norm
    from scipy.optimize import fsolve
    print("p=",norm.cdf(5,2,3)-norm.cdf(1,2,3))
    f=lambda c: norm.cdf(2*c,2,3)-norm.cdf(-c,2,3)-0.8
    print("c=",fsolve(f,0)[0])
    print('*'*100)


    #4.2计算正态分布概率密度函数
    import sympy as sp
    from sympy.core.numbers import pi
    x,y=sp.symbols('x,y')
    f=sp.solve(y-5/9*(x-32),x)[0]
    df=sp.diff(f,y)
    y1=1/sp.sqrt(4*pi)*sp.exp(-(x-98.6)**2/4)
    y2=y1.subs(x,f)*df
    print(sp.simplify(y2))  #得到的结果是以小数表示
    print('*'*100)

    #5.多维随机变量的计算：参见程序9~13
    # D:\self-learning\《Python工程数学应用》程序和数据\第3章  概率论与数理统计\

    #6.大数定理和中心极限定理：见程序17~18
    # D:\self-learning\《Python工程数学应用》程序和数据\第3章  概率论与数理统计\



    #7.随机变量的数字特征(均值、方差、偏度、峰度)
    '''
    【统计量】
    表示位置：算术平均值、中位数
    表示变异程度：标准差、方差和极差
    表示分布形状的统计量：偏度和峰度
    协方差和相关系数
    【统计图】
    ·频数表与直方图
    ·箱线图
    ·经验分布函数
    ·QQ图
    
    '''
    '''
    偏度和峰度定义:       https://zhuanlan.zhihu.com/p/84614017
    偏度和峰度如何影响分布: https://support.minitab.com/zh-cn/minitab/18/help-and-how-to/statistics/basic-statistics/supporting-topics/data-concepts/how-skewness-and-kurtosis-affect-your-distribution/
    '''
    #7.1 均值、方差、偏度和峰度
    from scipy.stats import binom
    n, p=10, 0.2
    mean, variance, skewness, kurtosis=binom.stats(n, p, moments='mvsk')
    print("所求的均值、方差、偏度和峰度分别为：",
          mean,' | ', variance,' | ', skewness,' | ', kurtosis)
    print('*'*50+'统计量分析'+'*'*50)


    #7.2通过pandas计算各种统计量
    import pandas as pd

    a = pd.read_csv('sanguo_data.csv', usecols=range(2, 8), header=0)
    print(a.head())  # 显示前5个任务的数据
    print('\n描述一下数据:')
    print(a.describe())
    print('\n------------------------------pandas计算各统计量------------------------------')
    MODE = a.mode()  # 计算各指标的众数
    SKEW = a.skew()  # 计算各指标的偏度
    KURT = a.kurt()  # 计算各指标的峰度
    COV = a.cov()  # 计算6个指标变量间的协方差矩阵
    CORR = a.corr()  # 计算6个指标变量间的相关系数矩阵
    print('\n众数:')
    print(MODE)
    print('\n偏度:')
    print(SKEW)
    print('\n峰度:')
    print(KURT)
    print('\n协方差矩阵:')
    print(COV)
    print('\n相关系数矩阵:')
    print(CORR)

    #7.3 通过numpy计算各种统计量
    print('\n------------------------------numpy计算各统计量------------------------------')
    b = a.values  # 提取其中的数值矩阵
    MEAN = b.mean(axis=0)  # 计算各指标的均值
    STD = b.std(axis=0)  # 计算各指标的标准差
    PTP = b.ptp(axis=0)  # 计算各指标的极差
    COV = np.cov(b.T)  # 计算6个指标变量间的协方差矩阵
    CORR = np.corrcoef(b.T)  # 计算6个指标变量间的相关系数矩阵
    print('\n均值:')
    print(a.columns.values)
    print(MEAN)
    print('\n标准差:')
    print(a.columns.values)
    print(STD)
    print('\n极差:')
    print(a.columns.values)
    print(PTP)
    print('\n协方差矩阵:')
    print(a.columns.values)
    print(COV)
    print('\n相关系数矩阵:')
    print(a.columns.values)
    print(CORR)
    print('*'*100)

    #8 常用统计图表
    #8.1 绘制直方图
    plt.rc('font', size=15);
    plt.rc('font', family="SimHei")
    # 下面提取6个指标变量的取值
    a = pd.read_csv('sanguo_data.csv', usecols=range(6, 8), header=0)
    plt.subplot(121);
    a["魅力"].hist(bins=10)  # 只画直方图
    plt.xlabel("魅力");
    plt.subplot(122)
    h = plt.hist(a["寿命"], bins=8)  # 画图并返回频数表
    plt.xlabel("寿命");
    print('返回的直方图纵坐标:',h[0])
    print('返回的直方图横坐标:',h[1])
    print('*'*100)
    # plt.show()


    #8.2绘制六个指标箱线图
    plt.rc('font', size=14)
    plt.rc('font', family='SimHei')
    a = pd.read_csv('sanguo_data.csv', header=0, usecols=range(2, 8))
    ax = a.boxplot();
    ax.grid()
    # plt.show()

    #8.3绘制经验分布函数
    plt.rc('font',size=14); plt.rc('font',family='SimHei')
    plt.rc('text',usetex=True)
    a=pd.read_csv('sanguo_data.csv',header=0,usecols=[6])
    b=a.values.flatten(); h=np.unique(b) #提出互异的点
    print('n=',len(h))  #显示不同的观测值个数
    ecdf=ECDF(b); Fh=ecdf(h)
    print(f'{len(h)}个经验分布函数值分别为:')
    print(Fh)  #显示经验分布函数值
    f=plt.hist(b,density=True, histtype='step', cumulative=True)
    plt.xlabel("$h$"); plt.ylabel("$F(h)$")
    print('*'*100)
    # plt.show()


    #8.4绘制QQ图
    plt.rc('font', size=14)
    plt.rc('font', family='SimHei')
    a = pd.read_csv('sanguo_data.csv', header=0, usecols=[6])
    b = a.values.flatten();
    n = len(a)
    mu = b.mean()
    s = b.std()  # 计算均值和标准差
    sx = sorted(b)  # 从小到大排列
    yi = norm.ppf((np.arange(n) + 1 / 2) / n, mu, s)
    plt.plot(yi, sx, '.', label='QQ图')
    plt.plot([1, 115], [1, 115], 'r-', label='参照直线')
    plt.legend()
    # plt.show()

    #9.区间估计
    '''
    区间估计总结:https://blog.csdn.net/weixin_43992800/article/details/100576931
    '''
    #9.1 正态分布求μ置信区间
    x0 = np.array([129, 134, 114, 120, 116, 133, 142, 138, 148, 129, 133, 141, 142])
    mu = np.mean(x0);
    s = np.std(x0, ddof=1)
    n = len(x0)
    t = st.t.ppf(0.975, n - 1)
    L = [mu - s / np.sqrt(n) * t, mu + s / np.sqrt(n) * t]
    print("μ置信区间为：", L)
    print('*'*100)

    #9.2 正态分布求σ置信区间
    x0 = np.array([506,508,499,503,504,510,497,512,514,505,493,496,506,502,509,496])
    s = np.std(x0, ddof=1)  #ddof=1代表方差无偏估计，除(n-1),没有默认除n
    n = len(x0)
    c1 = st.chi2.ppf(0.975, n - 1)
    c2 = st.chi2.ppf(0.025, n - 1)
    L = [np.sqrt(n - 1) * s / np.sqrt(c1), np.sqrt(n - 1) * s / np.sqrt(c2)]
    print("σ置信区间为：", L)
    print('*'*100)

    #10.方差分析
    '''
    若检验统计量F观察值>临界值，则拒绝原假设H0
    若检验统计量F观察值<临界值，则接受原假设H0
    
    详细步骤参考：
    https://mp.weixin.qq.com/s?__biz=MjM5MTI5MDgxOA==&mid=2650096293&idx=2&sn=78d0c2e989725f9b37a832f1227c2332&scene=21#wechat_redirect
    '''
    from scipy.stats import f
    a=np.array([[98,93,103,92,110],
      [100,108,118,99,111],[129,140,108,105,116]])  #使用时只需要更改a矩阵就行
    n = a.shape[0] * a.shape[1]
    s = a.shape[0]
    T = a.sum()  # 求所有元素的和
    n = a.size  # 元素的总个数
    St = (a ** 2).sum() - T ** 2 / n
    Tj = a.sum(axis=1)  # 计算行和
    nj = a.shape[1]  # 每行元素的个数
    Se = (a ** 2).sum() - (Tj ** 2 / nj).sum()
    Sa = St - Se
    F = (Sa / (s - 1)) / (Se / (n - s))  # 计算F统计量
    F0 = f.ppf(0.95, (s - 1), (n - s))
    print("F观察值为：", F)
    print("临界值为 ：", F0)
    if F > F0:
        print('拒绝原假设H0')
    else:
        print('接受原假设H0')
    print('*' * 100)


    # 11.回归分析
    # 11.1 线性回归

    a=np.array([[1000,600,1200,500,300,400,1300,1100,1300,300.],
             [   5,  7,  6,  6,  8,  7,  5,  4,  3,  9.],
             [ 100, 75, 80, 70, 50, 65, 90,100,110, 60.]])

    a=a.T       #加载表中的数据并转置
    md=LinearRegression().fit(a[:,:2],a[:,2])    #构建并拟合模型
    y=md.predict(a[:,:2])       #求预测值
    b0=md.intercept_    #输出截距，默认有截距
    b12=md.coef_        #输出回归系数
    R2=md.score(a[:,:2],a[:,2])      #计算R^2
    print('回归系数为：')
    print("b0=%.4f\nb12=%.4f%10.4f"%(b0,*b12))
    print('回归方程为：')
    each = [f"{b12[i]}x{i + 1}" for i in range(len(b12))]
    final_expression = "y=%.4f" % b0
    for i in each:
        if i[0] != '-':
            i = '+' + i
        final_expression += i
    print(final_expression)
    print("相关系数平方R^2=%.4f"%R2)
    print('*' * 100)

    # 11.2 多元多项式回归：包含线性、纯二次、交叉、完全二次
    from calculate_R_square import goodness_of_fit
    a = np.array([[1000, 600, 1200, 500, 300, 400, 1300, 1100, 1300, 300.],
                  [5, 7, 6, 6, 8, 7, 5, 4, 3, 9.],
                  [100, 75, 80, 70, 50, 65, 90, 100, 110, 60.]])

    a = a.T  # 加载表中的数据并转置
    y=a[:,2]
    md=LinearRegression().fit(a[:,:2],y)    #构建并拟合模型
    y1=md.predict(a[:,:2])                  #求预测值
    n=len(y1)
    rmse1=np.sqrt(((a[:,-1]-y1)**2).sum()/(n-3))
    R1_square=goodness_of_fit(y,y1)

    A2=np.hstack([a[:,:2],a[:,:2]**2])  #纯二次数据
    md2=LinearRegression().fit(A2,y)  #拟合纯二次型模型
    y2=md2.predict(A2)
    rmse2=np.sqrt(((a[:,-1]-y2)**2).sum()/(n-5))
    R2_square=goodness_of_fit(y,y2)

    A3=np.hstack([a[:,:2],(a[:,0]*a                [:,1]).reshape(-1,1)])  #交叉项数据
    md3=LinearRegression().fit(A3,y)  #拟合交叉项模型
    y3=md3.predict(A3)
    rmse3=np.sqrt(((a[:,-1]-y3)**2).sum()/(n-4))
    R3_square=goodness_of_fit(y,y3)

    A4=np.hstack([A3,a[:,:2]**2])  #完全二次数据
    md4=LinearRegression().fit(A4,y)  #拟合完全二次模型
    y4=md4.predict(A4)
    rmse4=np.sqrt(((a[:,-1]-y4)**2).sum()/(n-6))
    R4_square=goodness_of_fit(y,y4)

    print('剩余标准差RMSE:')
    print(['线性','纯二次','交叉项','完全二次'])  #显示所有剩余标准差
    print([rmse1,rmse2,rmse3,rmse4])  #显示所有剩余标准差

    print('\n拟合优度R²:')
    print(['线性','纯二次','交叉项','完全二次'])  #显示所有剩余标准差
    print([R1_square,R2_square,R3_square,R4_square])  #显示所有剩余标准差

    b0=md4.intercept_
    b=md4.coef_   #输出回归系数
    print('最好拟合情况系数(手动修改程序)：')
    print("b0=%.4f\nb=%.4f%10.4f%10.4f%10.4f%10.4f"%(b0,*b))    #分别对应截距,x1,x2,x1^2,x1x2,x2^2
    print('*'*100)


    #11.3 非线性回归
    #本例子拟合的是y=(β1x2)/(β1+β2x1+β3x2)
    from scipy.optimize import curve_fit
    a=np.loadtxt("Pgdata3_43.txt")   #加载表中的数据
    xy0=a[:2,:]; z0=a[2]
    def Pfun(t, b1, b2, b3):
        return b1*t[1]/(b1+b2*t[0]+b3*t[1])
    popt=curve_fit(Pfun, xy0, z0)[0]
    print("b1，b2，b3的拟合值                                                                                       为：", popt)
    print("预测值为：",Pfun([800,6],*popt))
    print('*'*100)







