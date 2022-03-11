import numpy as np
import scipy.sparse.linalg as SLA
a=np.array([[1,1,1,4,1,1/2],[1,1,2,4,1,1/2],[1,1/2,1,5,3,1/2],
   [1/4,1/4,1/5,1,1/3,1/3],[1,1,1/3,3,1,1],[2,2,2,3,1,1]])
[val,vec]=SLA.eigs(a,1)  #求最大模的特征值及对应的特征向量
B1=(vec/sum(vec)).real         #特征向量归一化
a1=np.array([[1,1/4,1/2],[4,1,3],[2,1/3,1]])  #健康情况的判断矩阵
a2=np.array([[1,1/4,1/5],[4,1,1/2],[5,2,1]])  #业务知识的判断矩阵
a3=np.array([[1,3,1/3],[1/3,1,1],[3,1,1]])    #写作能力的判断矩阵
a4=np.array([[1,1/3,5],[3,1,7],[1/5,1/7,1]])  #口才的判断矩阵
a5=np.array([[1,1,7],[1,1,7],[1/7,1/7,1]])   #政策水平的判断矩阵
a6=np.array([[1,7,9],[1/7,1,5],[1/9,1/5,1]])  #工作作风的判断矩阵
lamda=[]; B2=np.zeros((3,6))  #初始化
for i in range(1,7):
    s='a'+str(i)
    [v,vect]=SLA.eigs(eval(s),1)
    lamda.append(v)
    vect=vect.real; vect=vect/sum(vect)  #特征向量取实部并归一化
    B2[:,i-1]=vect.flatten()
B3=B2.dot(B1)   #求各对象的评价值
print(B1); print(lamda); print(B2); print(B3)
