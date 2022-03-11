import numpy as np
import matplotlib.pyplot as plt
#经过一下午漫长的探究才发现这两个函数计算的结果是一样的而且都是正确的
# # #################################拟合优度R^2的计算######################################
def __sst(y_no_fitting):
    """
    计算SST(total sum of squares) 总平方和
    :param y_no_predicted: List[int] or array[int] 待拟合的y
    :return: 总平方和SST
    """
    y_mean = sum(y_no_fitting) / len(y_no_fitting)
    s_list =[(y - y_mean)**2 for y in y_no_fitting]
    sst = sum(s_list)
    return sst


def __ssr(y_no_fitting,y_fitting):
    """
    计算SSR(regression sum of squares) 回归平方和
    :param y_fitting: List[int] or array[int]  拟合好的y值
    :param y_no_fitting: List[int] or array[int] 待拟合y值
    :return: 回归平方和SSR
    """
    y_mean = sum(y_no_fitting) / len(y_no_fitting)
    s_list =[(y_fit - y_mean)**2 for y_fit in y_fitting]
    ssr = sum(s_list)
    return ssr


def __sse(y_no_fitting,y_fitting):
    """
    计算SSE(error sum of squares) 残差平方和
    :param y_fitting: List[int] or array[int] 拟合好的y值
    :param y_no_fitting: List[int] or array[int] 待拟合y值
    :return: 残差平方和SSE
    """
    s_list = [(y_fitting[i] - y_no_fitting[i])**2 for i in range(len(y_fitting))]
    sse = sum(s_list)
    return sse


def goodness_of_fit(y_no_fitting,y_fitting):
    """
    计算拟合优度R^2
    :param y_no_fitting: List[int] or array[int] 待拟合y值
    :param y_fitting: List[int] or array[int] 拟合好的y值
    :return: 拟合优度R^2
    """
    SSR = __ssr(y_no_fitting,y_fitting )
    SST = __sst(y_no_fitting)
    rr = SSR /SST
    return rr


def get_lr_stats(y, y_prd):

    Regression = sum((y_prd - np.mean(y)) ** 2)  # 回归平方和
    Residual = sum((y - y_prd) ** 2)  # 残差平方和
    total = sum((y - np.mean(y)) ** 2)  # 总体平方和
    R_square = 1 - Residual / total  # 相关性系数R^2
    message1 = ('相关系数(R^2)： ' + str(R_square) + '；' + '\n' + '总体平方和(SST)： ' + str(total) + '；' + '\n')
    message2 = ('回归平方和(SSR)： ' + str(Regression) + '；' + '\n残差平方和(SSE)： ' + str(Residual) + '；' + '\n')
    message3 = ('SSR/SST ' + str(Regression/total) + '；' +'\n1-SSE/SST ' + str(1-Residual/total) + '；')
    return print(message1 + message2+message3+'\n'+'*'*100)

if __name__ == '__main__':
    # 生成待拟合数据
    a = np.arange(10)
    # 通过添加正态噪声，创造拟合好的数据
    b = a + 0.04 * np.random.normal(size=len(a))    #如果使用纯随机的数据，那么本身就不是拟合后的结果，得到的拟合优度大于1也是正常的，因为正常拟合出来的结果不可能会大于一
    print("原始数据为: ", a)
    print("拟合数据为: ", b)
    print('*'*100)
    print('按照函数1计算拟合优度：')

    rr = goodness_of_fit(a,b)
    SST=__sst(a)
    SSR=__ssr(a,b)
    SSE=__sse(a,b)
    print('SST=',SST)
    print('SSR=',SSR)
    print('SSE=',SSE)
    print('SSR/SST=',SSR/SST)
    print('1-SSE/SST=',1-SSE/SST)
    print(SST==SSE+SSR)
    print('*'*100)
    print('按照函数2计算拟合优度：')
    get_lr_stats(a,b)
    print("拟合优度为:", rr)
    plt.plot(a, a, color="#72CD28", label='原始数据')
    plt.plot(a, b, color="#EBBD43", label='拟合数据')
    plt.legend()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # plt.savefig(r"")
    plt.show()





