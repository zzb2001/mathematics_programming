'''
**************************************矩阵分析部分*******************************************
共轭相关可参考：https://blog.csdn.net/Strive_For_Future/article/details/108274742

'''

import numpy as np
from numpy import linalg as LA
from linear_algebra import eigenvals_and_eigenvects
from numpy import ndarray


'''函数都在这里定义'''
class array(ndarray):
    '''定义一个共轭转置H方法'''
    @property
    def H(self):
        return self.conj().T

if __name__ == '__main__':

    print('*'*50+'矩阵分析部分'+'*'*50)

    #1.矩阵的1-范数（列和范数）、∞-范数（行和范数）、Frobenius范数、2-范数（谱范数）求解
    a=np.array([[5,2,-2],[-1,4,3],[2,6,5]])
    L1=LA.norm(a,1); print('矩阵A的1-范数:',L1)           #1-范数
    Linf=LA.norm(a,np.inf); print('矩阵A的∞-范数:',Linf)  #inf-范数
    L3=LA.norm(a,'fro'); print('矩阵A的F-范数:',L3)       #Frobenius范数
    B=a.T.dot(a); val,vec=LA.eig(B)
    L4=np.sqrt(val.max()); print('矩阵A的2-范数:',L4)     #2-范数
    print('*'*100)

    #2.矩阵的奇异值分解流程：

    '''
    参考网址：https://zhuanlan.zhihu.com/p/29846048
    1.由于A是mxn不是方阵->A^H*A是nxn方阵->计算A^H*A的n个特征值与n个特征向量
    2.将计算出来的n个特征向量张成nxn的矩阵，该矩阵就是Q(还不是Q^H)
    3.A*A^H是mxm方阵->计算A*A^H的m个特征值与m个特征向量
    4.将计算出来的m个特征向量张成mxm的矩阵，该矩阵就是P
    5.根据σ_i=sqrt(λ_i)直接求出奇异值矩阵Σ
    注意：P、Q均为列正交矩阵，Q^H是行正交矩阵
    '''

    a=np.array([[1,0,1],[0,1,1],[0,0, 0]]).view(array)
    a_H=a.H
    aa_H=a.dot(a_H)
    eigenvals_and_eigenvects(aa_H)  #计算A^H*A的特征值与特征向量
    print('矩阵的秩为：', np.linalg.matrix_rank(a))#计算A的秩
    p,d,q=LA.svd(a)   #a=p*d*q
    print('P:',p);
    print('D:',d);
    print('Q^H:',q)

    #3.奇异值分解应用：
    '''
    1.图像压缩，参见程序11：
    D:\self-learning\《Python工程数学应用》程序和数据\第2章  矩阵分析
    
    2.对应分析(R型因子分析(nxn)、Q型因子分析(mxm))，参见程序12：
    D:\self-learning\《Python工程数学应用》程序和数据\第2章  矩阵分析
    参考网址：
    https://blog.csdn.net/mengjizhiyou/article/details/83243248
    
    3.语义挖掘，参见程序13：
    D:\self-learning\《Python工程数学应用》程序和数据\第2章  矩阵分析

    '''
