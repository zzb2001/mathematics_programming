'''
**************************************Numpy基础*******************************************
'''
import numpy as np
import time
if __name__ == '__main__':
    print('*'*30+'Numpy基础'+'*'*30)

    '''1.Numpy
    1.1.数组的创建：各种随机矩阵、特殊矩阵生成
    1.2.数组属性：维数ndim、shape(m,n)、size(m*n)、数组类型dtype、每个元素大小itemsize
    1.3.数组变形
    1.4.数组索引：整形索引、布尔索引、花式(一维/二维数组作为索引)索引
    1.5 矩阵mat、Matrix（前面二者等价）、bmat：三种方法
    1.6Numpy中array与Matrix比较：
        ①乘法
        array： *表示元素乘法，dot()表示矩阵乘法，@表示矩阵乘法
        Matrix：*表示矩阵乘法，multiply()表示元素乘法，@表示矩阵乘法
        ②向量转置
        array:(n,)表示行向量/列向量，转置不变；dot(A,v)看作列，dot(v,A)看作行;只有加上了1之后才确定行/列
        Matrix:严格行列
        ③属性
        array:.T表示转置
        Matrix:.H表示共轭转置，.I表示求逆矩阵，.A表示转化成array数组
    1.7Numpy与Matlab比较：见资料

    '''

    # 1.3.数组变形：矩阵横向拼接
    print('*'*50+'矩阵横向拼接'+'*'*50)
    a=np.ones((4,1))
    b=np.diag(np.array([2,3,1.5]),1)
    c=np.eye(4,3,k=-1)
    d1=np.hstack((a,b,c))  #第一种方法
    print('拼接后的矩阵：',d1)
    d2=np.c_[a,b,c]        #第二种方法
    print('拼接后的矩阵：',d2)
    time.sleep(0.2)

    #1.4.数组索引
    #1.4.1整数与切片索引
    print('*'*50+'整数与切片索引'+'*'*50)
    a = np.arange(1, 19).reshape(3, 6);
    print(a)
    print(a[1, 2], ',', a[1][2])  # a[1,2]与a[1][2]两者相同
    print(a[1:2, 2:3])  # 通过切片得到二维数组，切片把每一行每一列当成一个列表
    print(a[1:, :-1])  # 去掉第一行和最后一列得到的二维数组
    print(a[-1])  # 提取最后一行得到的一维数组
    time.sleep(0.2)

    # 1.4.2 布尔索引
    print('*' * 50 + '布尔索引取值' + '*' * 50)
    a = np.arange(1, 17).reshape(4, 4)
    print('原矩阵：')
    print(a)
    b = np.array([1, 2, 6, 2])
    c = a[b == 2]                   # 提取a数组的第2、4行
    print(b == 2, '\n', c)
    time.sleep(0.2)

    # 1.4.3 布尔索引取否
    print('*' * 50 + '布尔索引取否' + '*' * 50)
    a = np.arange(1, 17).reshape(4, 4)
    print('原矩阵：')
    print(a, '\n----------')
    b = np.array([1, 2, 6, 2])
    c1 = a[~(b == 2)]               # 提取a数组第1、3行的第一种方法
    c2 = a[b != 2]                  # 提取a数组第1、3行的第二种方法
    c3 = a[np.logical_not(b == 2)]  # 提取a数组第1、3行的第三种方法
    print(c1, '\n----------\n', c2, '\n----------\n', c3)
    time.sleep(0.2)


    #1.4.4花式索引
    #一维数组作为索引，索引结果就是对应位置的元素
    #二维数组作为索引（两维度相同的一维数组），索引结果就是对应横纵坐标位置索引的值组成的新一维数组
    print('*' * 50 + '花式索引' + '*' * 50)
    from numpy import arange, array
    x = arange(6)
    print("前三个元素为：", x[[0, 1, 2]])  # 显示[0 1 2]
    print("后三个元素为:", x[[-1, -2, -3]])  # 显示[5,4,3]
    y = array([[1, 2], [3, 4], [5, 6]])
    print("前两行元素为：\n", y[[0, 1]])  # 显示前两行[[1,2],[3,4]]
    print('------\n', y[[0, 1], [0, 1]])  # 显示y[0][0]和y[1][1]组成的一维数组
    time.sleep(0.2)


    #1.5矩阵对象
    print('*' * 50 + '矩阵对象' + '*' * 50)
    #方法一：分号隔开的字符串创建矩阵
    a = np.mat("1 2 3;4 5 6;7 8 9")  # 或者写作a=np.mat('1,2,3;4,5,6;7,8,9')
    print("{}\n{}".format(a, type(a)))
    #方法二：通过numpy.array对象创建
    a = np.arange(1, 10).reshape(3, 3)  # 创建数组
    b = np.mat(a);  # 创建矩阵
    print("a={}\nb={}".format(a, b))
    #方法三：通过bmat创建分块矩阵
    A = np.eye(2);
    B = 3 * A
    C = np.bmat([[A, B], [B, A], [A, B]])  # 构造6行4列的矩阵
    print("C=\n{}\n 维数为：{}".format(C, C.shape))
    time.sleep(0.2)






