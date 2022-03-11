'''
**************************************线性代数部分*******************************************
sympy学习网址：
https://blog.csdn.net/cj151525/article/details/95756847
'''

import numpy as np
import scipy
import sympy as sp
import pylab as plt

'''函数都在这里定义，不然其他程序在调用你这个函数的时候会将这里面的东西全部打印出来'''
def get_inverse_Matrix(a):
    ainv=np.linalg.inv(a)
    return ainv
def get_reverse_order_number(a):
    '''
    输入一个np.array对象，输出逆序数的值
    思路：遍历至第i个数(1<=i<=n)，找到前面有多少个大于他的数然后求和
    '''
    n=len(a)
    s=0
    for i in np.arange(1,n):
        ind=np.where(a[:i]>a[i])
        s+=len(ind[0])
    print('逆序数为：',s)
    return s

def get_adjoint_matrix(A,mode=1):
    '''输入一个np.array,输出一个np.array'''
    if mode==0:
        n = len(A)
        B = np.zeros((4, 4))
        for i in range(n):
            for j in range(n):  # 伴随矩阵需要求出余子式,余子式本质是行列式,只有方阵才能求行列式
                Hij=A.copy()
                Hij=np.delete(Hij,i,axis=0) #把对应的i，j两列删了算余子式
                Hij=np.delete(Hij,j,axis=1)
                B[j,i]=(-1)**(i+j)*np.linalg.det(Hij)
                # B[i,j]=(-1)**(i+j)*np.linalg.det(Hij)
                # B=B.T
        print('余子式法所求的伴随矩阵为：')
        print(B)
        return B
    elif mode==1:
        # 法二：利用AA^*=A^*A=|A|E
        A = np.array([[3, 1, -1, 2], [-5, 1, 3, -4], [2, 0, 1, -1], [1, -5, 3, -3]])
        B = np.linalg.det(A) * np.linalg.inv(A)
        print('公式法所求的伴随矩阵为：')
        print(B)
        return B
    elif mode==2:
        # 法三：哈密顿-凯莱定理(Hamilton-Cayley theorem)
        A = np.mat(
            [[3, 1, -1, 2], [-5, 1, 3, -4], [2, 0, 1, -1], [1, -5, 3, -3]])  # 注意：这个地方只能用np.mat,如果使用array得到的结果会不一样
        P1 = np.poly(A)
        P2 = P1[:-1]
        B = (-1) ** (len(A) - 1) * sum([P2[i] * A ** (3 - i) for i in range(4)])
        print('哈密顿-凯莱定理所求的伴随矩阵为：')
        print(B)
        return B

def eigenvals_and_eigenvects(A):
    if isinstance(A,np.ndarray):
        ##求数值解
        p=np.poly(A)                #计算特征多项式
        p_formula=np.poly1d(p)      #这可以将多项式系数p打印出来
        w1=np.roots(p)              #求特征根
        w2,v=np.linalg.eig(A)       #w为特征值，v为特征向量
        w3=np.linalg.eigvals(A)     #求特征值
        print('矩阵的特征多项式为:\n',p_formula)
        print('[方式一]矩阵的特征值为:',w1)
        print('[方式二]矩阵的特征值为:',w2)
        print('[方式三]矩阵的特征值为:',w3)      #这三种方法都是计算特征值的
        print('矩阵的特征向量为:\n',v)
        print('*'*100)
        return w2,v
    else:
        ##求符号解
        lamda=sp.symbols('lamda')
        p=A.charpoly(lamda)     #计算特征多项式,这样可以把完整的表达式输出来
        w1=sp.roots(p)          #计算特征根
        w2=A.eigenvals()        #直接计算特征值
        v=A.eigenvects()        #直接计算特征向量
        print('特征多项式为:',p)
        print("特征值为：",w2)   #{2: 1, 1: 2}表示2有1重，1有2重
        print("特征向量为：")     #(特征值：代数多重性，[特征向量])
        for each in v:
            print(each)
        print('*'*100)
        return w2,v

def diag(A):
    if A.is_diagonalizable():
        P,D=A.diagonalize() #可逆矩阵P,使得P^(-1)AP=D
        print('A可以对角化！')
        print("P=",P); print("D=",D)
        print('*'*100)
        return P,D
    else:
        print("A不能对角化")
        print('*'*100)

def quadratic_form(A):
    '''输入一个sp.Matrix,输出正交变换矩阵P和二次型对角矩阵D'''
    P1,D=A.diagonalize()  #把A对角化
    P2=np.array(P1); P2=[sp.Matrix(p) for p in P2]
    P3=sp.GramSchmidt(P2,True)  #施密特正交化、单位化
    print('正交矩阵：')
    for each in P3:
        print(each)
    print('化为标准型的对角矩阵：',D)
    print('*'*100)
    return P3,D

def is_positive(A,mode=1):
    '''输入np.array'''
    if mode==1:
        b = np.zeros(3)  # 存放顺序主子式数组的初始化
        for i in range(len(A)):
            b[i] = np.linalg.det(A[:i + 1, :i + 1])
        print('A矩阵的顺序主子式为:', b)  # 显示各阶顺序主子式的取值
        if all((b > 0)):
            print('A为正定矩阵')
        elif any((b < 0)):
            print('A为负定矩阵')
        print('*'*100)
    elif mode==2:
        vals = np.linalg.eigvals(A)  # 求a的特征值
        print("A矩阵所有的特征值为:\n", vals)
        if np.all(vals > 0):
            print("A是正定的")
        elif np.all(vals < 0):
            print("A是负定的")
        else:
            print("A非正定也非负定")
        print('*'*100)


if __name__ == '__main__':

    print('*'*50+'行列式部分'+'*'*50)
    
    #求行列式逆序数
    a=np.array([3,1,5,4,2])
    get_reverse_order_number(a)
    print('*'*100)


    '''行列式'''
    #数值计算:只能算出解的近似值，得不到准确结果
    a=np.ones((4,4))
    for i in range(a.shape[0]):
        a[i,i]=3
    D=np.linalg.det(a)
    print('行列式计算结果为D=',round(D,2))

    #符号计算：可以得到准确结果（例如计算pi的时候能准确的算出sin(pi)=0）
    a=sp.ones(4)
    for i in range(a.shape[0]):
        a[i,i]=3
    D=a.det()   #determinant
    print('行列式计算结果为D=',D)

    #行列式的符号运算1
    a,b,c,d=sp.symbols('a:d')
    A=sp.Matrix([[a,b,c,d],[0,a,a+b,a+b+c],[0,a,2*a+b,3*a+2*b+c],[0,a,3*a+b,6*a+3*b+c]])
    D=sp.det(A)
    print('行列式计算结果为D=',D)

    #行列式的符号运算2
    x=sp.symbols('x')
    a=sp.symbols('a0:4')
    A=sp.eye(4)*x
    A[-1,:]=sp.Matrix([a])
    B=np.diag(-np.ones(3),1).astype(int)
    A=A+sp.Matrix(B)
    a=A.det()
    print('行列式计算结果为A=',a)
    print('*'*100)

    #克拉默法则解齐次方程:齐次方程有齐次解，则证明其系数行列式D=0
    l=sp.symbols('lambda')
    A=sp.Matrix([[5-l,2,2],[2,6-l,0],[2,0,4-l]])
    D=sp.det(A)
    DD=sp.factor(D) ##公式折叠用factor方法，展开用expand方法
    s=sp.solve(DD)
    for i in range(0,len(s)):
        print(f'齐次方程第{i+1}个解为：',s[i])


    '''
    矩阵运算与线性变换
    np.flip(A,axis=0)   #axis=0表示关于x轴翻转，axis=1表示关于y轴翻转
    np.fliplr(A)        #左右翻转
    np.flipud(A)        #上下翻转
    np.rot90(A)         #逆时针旋转90°
    np.tril(A)          #提取矩阵下三角
    np.triu(A)          #提取矩阵上三角
    np.reshape(A,(m,n)) #变形
    np.tile(A,m)        #A作为子块生成1xm的分块矩阵
    np.tile(A,(m,n))    #A作为子块生成mxn的分块矩阵
    np.transpose(A)     #矩阵A的转置矩阵
    np.dot(A,B)         #矩阵A与B相乘
    np.linalg.inv(A)    #逆矩阵   
    np.linalg.pinv(B)   #计算A的Moore_Penrose伪逆
    np.linalg.qr(A)     #计算A的QR值分解
    np.linalg.svd(A)    #计算A的奇异值分解
    '''


    print('*'*50+'矩阵部分'+'*'*50)

    #矩阵乘法
    A=sp.Matrix(3,4,sp.symbols('A1:4(1:5)'))    #下角标有两位
    B=sp.Matrix(4,2,sp.symbols('B1:5(1:3)'))
    C=A@B   #A*B
    print('矩阵乘法的结果为：')
    for i in range(len(C)):
        print(C[i])
    print('*'*100)



    #伴随矩阵
    #法一：利用代数余子式

    A=np.array([[3,1,-1,2],[-5,1,3,-4],[2,0,1,-1],[1,-5,3,-3]])
    get_adjoint_matrix(A,mode=1)
    print('*'*100)


    #求逆矩阵
    ##数组计算
    A=np.array([[1,2,3],[2,2,1],[3,4,3]])
    B=np.array([[2,1],[5,3]])
    C=np.array([[1,3],[2,0],[3,1]])
    Ainv=np.linalg.inv(A)
    Binv=np.linalg.inv(B)
    X=Ainv.dot(C).dot(Binv)
    print('逆矩阵计算结果：')
    print(X)
    print('*'*100)

    ##矩阵计算
    A=np.mat([[1,2,3],[2,2,1],[3,4,3]])
    B=np.mat([[2,1],[5,3]])
    C=np.mat([[1,3],[2,0],[3,1]])
    Ainv=np.linalg.inv(A)
    Binv=np.linalg.inv(B)
    X=Ainv@C@Binv   #X=Ainv*C*Binv
    print('逆矩阵计算结果：')
    print(X)
    print('*'*100)
    ##注意：sympy中*表示矩阵乘法，Numpy中*表示矩阵对应元素相乘


    #空间变换
    #对⚪：(x-1)^2+y^2=1求其关于y=3x+5的镜像曲线并画出图形,再通过反变换得到原来的解
    plt.rc('font',size=16)
    plt.rc('font',family='SimHei')
    plt.rc('axes',unicode_minus=False)
    x0,y0,X,Y,t=sp.symbols('x0,y0,X,Y,t')
    equ1=(Y+y0)/2-3* (X+x0)/2-5
    equ2=(Y-y0)*3+(X-x0)
    s=sp.solve([equ1,equ2],[X,Y])   #方程的解是一个列表
    XX=s[X]
    YY=s[Y]
    print('镜像后关于x0，y0的圆方程:','x1=',XX,'x2=',YY)
    X1=XX.subs({x0:1+sp.cos(t),y0:sp.sin(t)})   #把x0，y0通过⚪方程换掉
    Y1=YY.subs({x0:1+sp.cos(t),y0:sp.sin(t)})
    print('化简成仅有参数t:','1=',X1,'Y1=',Y1)

    t0=np.linspace(0,2*np.pi,100)
    x1=1+np.cos(t0)
    y1=np.sin(t0)
    plt.axes(aspect='equal')
    plt.plot(x1,y1)         #画原来的圆
    plt.text(-0.1,1.2,'原来的圆')
    x2=np.linspace(-3,0,10)
    y2=3*x2+5
    plt.plot(x2,y2)         #画直线
    x3=[X1.subs(t,v) for v in t0]
    y3=[Y1.subs(t,v) for v in t0]
    plt.plot(x3,y3)         #画新圆
    plt.text(-4.5,2.8,'镜像圆')
    # plt.show()

    #反变换
    T1=sp.Matrix([[-4/5,3/5,-3],[3/5,4/5,1],[0,0,1]])
    T2=T1.inv()
    XY=sp.Matrix([X1,Y1,1]) # XY=sp.Matrix([[X1],[Y1],[1]])也行
    xy=sp.simplify(T2*XY)
    xy=xy[:-1]
    print('变换回原来的圆方程：',xy[0],xy[1])
    print('*'*100)


    #矩阵的初等变换
    A=np.array([[2,-1,-1],[1,1,-2],[4,-6,2]],dtype=int)
    E=np.eye(3,dtype=int)
    AE=np.hstack([A,E])
    AE=sp.Matrix(AE)    #将array矩阵变成符号矩阵，这样才能使用sp的rref
    SAE=AE.rref()       #把AE矩阵化简成最简式。Matrix().rref()返回两个元素的元组。第一个是精简行梯形形式，第二个是枢轴列索引元组。
    F=SAE[0][:,:3]
    P=SAE[0][:,3:]
    print('矩阵A初等变换化成最简型F:')
    print(F)
    print('可逆矩阵P:')   #对于可逆矩阵P，有PA=F
    print(P)
    print('*'*100)

    #计算矩阵的秩
    A=np.array([[2,-1,-1],[1,1,-2],[4,-6,2]],dtype=int)
    print('矩阵的秩为：',np.linalg.matrix_rank(A))
    print('*'*100)


    #求齐次线性方程组的解：
    #化成最简型之后调用库函数求零空间直接得到最后的基础解系
    a=sp.Matrix([[1,-1,-1,1],[1,-1,1,-3],[1,-1,-2,3]])
    b=a.rref()
    print('齐次方程系数矩阵化简成最简形式：')
    print(b)  #化行最简形并显示
    x=a.nullspace()  #直接调用库函数求零空间，即基础解系
    print("基础解系为：\n",x)
    print('*'*100)


    #求非齐次线性方程组的解：
    #只能通过rref方法将增广矩阵化简成最简型,然后手动写成基础解系的形式
    a=np.array([[1,1,-3,-1],[3,-1,-3,4],[1,5,-9,-8]])
    b=np.array([[1,4,0]]).T
    c=np.hstack([a,b])
    d=sp.Matrix(c).rref()
    print('化简成最简型的增广矩阵为：')
    print(d)
    print('*'*100)


    #无论数学上Ax=b是否有解/多解/无解，x=pinv(a).dot(b)总是能给出唯一解
    #①无穷多解的时候给出的是最小范数解
    #②无解时给出的是最小二乘解，即方程两边误差平方和最小的解。这个一般用于【拟合与回归】
    a=np.array([[1,1,-2,-1],[1,5,-3,-2],[3,-1,1,4],[-2,2,1,-1]])
    b=np.array([[-1,0,2,1]]).T
    ab=np.hstack([a,b])
    r1=np.linalg.matrix_rank(a)   #计算系数矩阵的秩
    r2=np.linalg.matrix_rank(ab)  #计算增广矩阵的秩
    print("系数矩阵和增广矩阵的秩分别为：",r1,',',r2)
    x=np.linalg.pinv(a).dot(b)
    print("所求的最小范数解为:\n",x)
    print('*'*100)


    #拟合函数(一阶、二阶、对数)
    x0=np.arange(1,9)
    y0=np.array([8,12,7,14,15,16,18,21])
    xs=np.c_[x0,np.ones(8)]  #构造线性方程组的系数矩阵，8*2
    #np.r_是按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等。
    #np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等。

    ab1=np.linalg.pinv(xs).dot(y0)  #这相当于是把x1想成x1,b想成x2,（a,1）想成系数矩阵，通过矩阵乘法的方式来算
    ab2=np.polyfit(x0,y0,1)
    print("矩阵乘法拟合的参数为：",ab1)   #一阶拟合的时候两个参数的情况是一样的
    print("多项式拟合拟合的参数为：",ab2)   #一阶拟合的时候两个参数的情况是一样的
    cde=np.polyfit(x0,y0,2)  #拟合第二个函数的参数
    print("第二个函数拟合的参数为：\n",cde)
    xs3=np.c_[np.log(x0),1/x0]  #构造拟合f,g时线性方程组的系数矩阵
    fg=np.linalg.pinv(xs3).dot(y0)
    print("第三个函数拟合的参数为：\n",fg)
    print('*'*100)


    #利用稀疏矩阵和普通矩阵两种格式分别求解方程组
    from scipy.sparse.linalg import spsolve
    from scipy.sparse import csr_matrix
    import time
    a=4*np.eye(1000)+np.eye(1000,k=1)+np.eye(1000,k=-1)
    b=np.arange(1,1001)
    x1=np.linalg.inv(a).dot(b)
    start=time.time()  #1970纪元后经过的浮点秒数
    x2=np.linalg.solve(a,b)  #线性方程组的另一种普通解法
    T1=time.time()-start  #计算求解花费的时间
    a2=csr_matrix(a)    #创建稀疏矩阵
    start=time.time()
    x3=spsolve(a2,b)
    T2=time.time()-start
    cha=sum(abs(x2-x3))  #比较计算的误差
    print('稠密矩阵计算时间:',T1)  #显示稠密矩阵花费的时间
    print('稀疏矩阵计算时间:',T2)  #显示稀疏矩阵花费的时间
    print('*'*100)


    '''
    特征值与特征向量：
    A=np.array([[2,-1,-1],[1,1,-2],[4,-6,2]],dtype=int)
    p=np.poly(A)                #返回矩阵A的特征多项式,p为(n+1)维向量（从lambda^n到lambda^0）
    root=np.roots(p)            #计算特征根
    trace=np.trace(A)           #计算矩阵A对角线元素的和，也等于所有特征值的和
    w,v=np.linalg.eig(A)        #w为特征值，v为特征向量
    w=np.linalg.eigvals(A)      #求一般矩阵的特征值
    w,v=np.linalg.eigh(A)       #求实对称阵或者复Hermitian矩阵的特征值与特征向量,w为特征值，v为特征向量
    w=np.linalg.eigvalsh(A)     #求实对称阵或者复Hermitian矩阵的特征值
    '''

    #求特征值与特征向量
    A=np.array([[-1,1,0],[-4,3,0],[1,0,2]])
    eigenvals_and_eigenvects(A)
    A = sp.Matrix([[-1, 1, 0], [-4, 3, 0], [1, 0, 2]])
    eigenvals_and_eigenvects(A)



    '''相似矩阵与正交化'''
    #施密特正交化
    #输入一个线性无关向量组，导出一个正交向量组
    from sympy import Matrix, GramSchmidt
    A=[Matrix([1,2,-1]),Matrix([-1,3,1]),Matrix([4,-1,0])]  #A必须为列表
    B=GramSchmidt(A,True)
    print("所求的正交规范化向量组为：")
    for each in B:
        print(each)
    print('*'*100)


    #已知α1=[1,1,1]，求一组非零向量α2、α3，使得三个向量两两正交
    #提示：α2、α3都满足(α1.T)x=0,所以解这个齐次方程组，系数矩阵为[1,1,1]，解得基础解系之后再进行正交化
    from sympy import Matrix, GramSchmidt
    A1=Matrix([[1,1,1]])
    X=A1.nullspace()
    B=GramSchmidt(X)
    print("所求的正交向量为：")
    for each in B:
        print(each)
    print('*'*100)


    #能否对角化：
    #   特征向量个数=n  ==>可以
    #            !=n  ==>不可以
    #   相似对角化（实际为坐标轴旋转）可以消去二次型中的交叉项
    #更进一步理解可以参考：https://zhuanlan.zhihu.com/p/138285148

    A=sp.Matrix([[-1,1,0],[-4,3,0],[1,0,2]])        #不能对角化
    eigenvals_and_eigenvects(A)
    diag(A)
    A=sp.Matrix([[0,-1,1],[-1,0,1],[1,1,0]])        #可以对角化
    eigenvals_and_eigenvects(A)
    diag(A)
    A=sp.Matrix([[1, 2, 0], [0, 3, 0], [2, -4, 2]]) #可以对角化
    eigenvals_and_eigenvects(A)
    diag(A)

    #相似：如果有n阶可逆矩阵P存在，使得P^(-1)AP=B，则称矩阵A与B相似，记为A~B
    #性质：特征值相同:|A|=|B|=lambda1*lambda2*lambda3...
    #     秩相同
    #     迹相同:tr(A)=tr(B)=lambda1+lambda2+lambda3
    #     相同特征值对应的线性无关的特征向量个数相同.


    #问题：如果矩阵A与矩阵B相似，求x,y
    x,y=sp.symbols('x,y')
    A=sp.Matrix([[1,-2,-4],[-2,x,-2],[-4,-2,1]])
    B=sp.diag(5,-4,y)
    eq1=A.det()-B.det()
    eq2=A.trace()-B.trace()
    xy=sp.solve((eq1,eq2),(x,y))
    print("x的值为:",xy[x],'；',"y的值为：",xy[y])
    print('*'*100)


    #相似对角化是可对角化矩阵的方幂运算的工具
    A=sp.Matrix([[2,1,2],[1,2,2],[2,2,1]])
    P,D=A.diagonalize()  #把A相似对角化
    s1=P*(D**10-6*D**9+5*D**8)*(P.inv())
    A2=np.mat(A)
    s2=A2**10-6*A2**9+5*A**8  #直接计算矩阵多项式
    print('利用相似对角化简便计算后的结果:',s1)
    print('矩阵直接乘幂得到的结果:',s2)
    print('*'*100)



    '''二次型'''
    #求一个正交变换，使得二次型f=-2x1x2+2x1x3+2x2x3化为标准型


    A=sp.Matrix([[0,-1,1],[-1,0,1],[1,1,0]])
    P,D=quadratic_form(A)

    #判断正定性与负定性：①赫尔维兹定理：顺序主子式大于0  ②根据特征值的正负号判定二次型的正定性

    A = np.array([[-5, 2, 2], [2, -6, 0], [2, 0, -4]],dtype=int)
    is_positive(A,mode=1)
    is_positive(A,mode=2)

    #*最大模特征值以及对应的特征向量
    #应用于层次分析法、马尔科夫链以及PageRank算法中
    #二次型的最大值为矩阵A的最大特征值，也就是在能取得最大特征值对应的特征向量上取到，反之亦然
    A=np.array([[3,1,2],[1,6,-2],[2,-2,1]],dtype=np.float32)    #这里一定要使用浮点型，不然报错
    val,vec=np.linalg.eig(A)  #求特征值和特征向量
    print("矩阵A特征值为：",val)
    print("矩阵A特征向量为：\n",vec)    #注意，读这个3*3的矩阵是竖着读的
    # A=sp.Matrix([[3,1,2],[1,6,-2],[2,-2,1]])
    val,vec=scipy.sparse.linalg.eigs(A,1,which='LM')#LM指的是求前k个最大模的特征值和特征向量
    print("矩阵A最大特征值为：",val)
    print("矩阵A最大特征值对应的最大特征向量为：\n",vec)
    val,vec=scipy.sparse.linalg.eigs(A,1,which='SM') #SM指的是求后k个最大模的特征值和特征向量
    print("矩阵A最小特征值为：",val)
    print("矩阵A最小特征值对应的最大特征向量为：\n",vec)
    print('*'*100)


















