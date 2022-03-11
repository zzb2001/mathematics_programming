'''Scipy介绍'''
import numpy as np
from scipy.integrate import quad,dblquad
'''积分'''
#定积分
print('-'*30+'定积分计算'+'-'*30)
y=lambda x,a,b:np.sqrt(a*x**2+b*np.sin(x))
I,err=quad(y,0,1,args=(2,1))
print('积分值为：',I)
print('积分误差为：',err)
#计算曲面积分
print('-'*30+'曲面积分计算'+'-'*30)
I,err=dblquad(lambda x,y: x*y,    #被积函数
          -1,2,    #下界和上界
          lambda y:y**2, lambda y:y+2)
print('积分值为：',I)
print('积分误差为：',err)

'''求解非线性方程组'''
from scipy.optimize import fsolve
print('-'*30+'非线性方程组计算'+'-'*30)

def func(t):
    x,y,z=t
    return [x+2*y+3*z-6,
            5*x**2+6*y**2+7*z**2-18,
            9*x**3+10*y**3+11*z**3-30]
s=fsolve(func,np.random.rand(3))
print('方程的解为：',s)


'''求函数极值点和最值点'''
print('-'*30+'极值最值计算1'+'-'*30)
from scipy.optimize import fmin,fminbound,minimize,leastsq
import pylab as plt
yx=lambda x: x**2+10*np.sin(x)+1
x=np.linspace(-6,6,100)
x1=fmin(yx,5) #求5附近的极小点
print("极小点：",x1);
x2=fminbound(yx,-6,6) #区域的最小点
print("最小值为：",yx(x2))
plt.plot(x,yx(x));
plt.show()

print('-'*30+'极值最值计算2'+'-'*30)
def f(X):
    x,y=X
    return (x-1)**4+5*(y-1)**2-2*x*y
X=minimize(f,[0,0]).x
val=minimize(f,[0,0]).fun
print("极小点和极小值分别为：",X,',',val)
x=y=np.linspace(-1,4,100)
x,y=np.meshgrid(x,y)
c=plt.contour(x,y,f((x,y)),40)
plt.clabel(c); plt.colorbar()
plt.plot(X[0],X[0],'Pr'); plt.show()

'''最小二乘法'''
#f(x,y)=axy+bsin(cx),取a=2,b=3,c=4构造数据，利用模拟数据反过来拟合函数
from scipy.optimize import curve_fit
print('-'*30+'最小二乘法'+'-'*30)
x=y=np.linspace(-6,6,30)
fxy=lambda t,a,b,c: a*t[0]*t[1]+b*np.sin(c*t[0])
z=fxy([x,y],2,3,4)
p=curve_fit(fxy,[x,y],z,bounds=([1,2,3],[3,4,5]))[0]
print(p)
#非线性拟合，必须得约束拟合参数的上界与下界才能得到好的拟合结果

'''求微分方程的数值解'''
#求解下列微分方程组：
#x'=-x^3-y,x(0)=10
#y'=x-y^3,y(0)=0.5  0<=t<=30
from scipy.integrate import odeint
import pylab as plt
print('-'*30+'求解微分方程的数值解'+'-'*30)2
plt.rc('text',usetex=True)
plt.rc('font',family="SimHei")
def func(w,t):
    x,y=w;
    return [-x**3-y,x-y**3]
t=np.linspace(0,30,100)
s=odeint(func,[1,0.5],t)
plt.subplot(121)
plt.plot(t,s[:,0],'*-',label="$x(t)$")  #画出了x(t)的解曲线
plt.plot(t,s[:,1],'--p',label="$y(t)$") #画出了y(t)的解曲线
plt.legend()
plt.subplot(122)
plt.plot(s[:,0],s[:,1])     #画出解的轨迹线
plt.show()



