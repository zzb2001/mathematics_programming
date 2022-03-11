'''
**************************************积分部分*******************************************
概念速查：
'''

if __name__ == '__main__':
    print('*'*50+'积分部分'+'*'*50)

    '''傅里叶变换'''
    '''拉普拉斯变换'''
    #1.求时域函数的拉普拉斯变换
    import sympy as sp
    sp.var('t,s,k')  #定义符号变量
    sp.var('n',integer=True)  #定义整型符号变量
    L1=sp.laplapce_transform(s.DiracDelta(t),t,s)
    L2=sp.laplace_transform(1,t,s)
    L3=sp.laplace_transform(sp.exp(k*t),t,s)
    L4=sp.laplace_transform(sp.sin(k*t),t,s)
    L5=sp.laplace_transform(sp.cos(k*t),t,s)
    L6=sp.laplace_transform(t**n,t,s)
    print('L1=',L1); print('L2=',L2)
    print('L3=',L3); print('L4=',L4)
    print('L5=',L5); print('L6=',L6)
