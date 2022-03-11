import numpy as np; import pandas as pd
import sympy as sp
import scipy.sparse.linalg as SLA
a=pd.read_csv('Pgdata1_40.txt',header=None,sep='\t')
b=a.values            #提取所有的数据
c=b[1::2,:]           #提取市场状态数据
d=np.delete(c,-1).astype(int)  #删除最后一个元素
f=np.zeros((3,3),dtype=int)
for i in range(len(d)-1):f[d[i]-1][d[i+1]-1] +=1
P=f/np.tile(f.sum(axis=1).reshape(3,1),(1,3))
val,vec=SLA.eigs(P.T,1)  #求最大特征值1对应的特征向量
vec=vec.real; vec=vec/sum(vec)  #特征向量归一化
print(vec)
