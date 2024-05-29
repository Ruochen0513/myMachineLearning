import numpy as np
from matplotlib import pyplot as plt
import datetime
class SupportVectorMachine:
    def __init__(self, train_x, train_y, train_num):
        self.train_x = train_x
        self.train_y = train_y
        self.train_num = train_num
        self.max_iter = 100000
        self.alpha = np.zeros((train_num, 1))  # 拉格朗日乘子
        self.C = 1  # 惩罚系数
        self.beta = 0.1   # 拉格朗日系数
        self.lamda = 0.1  # 拉格朗日系数
        self.eta = 0.0001
        # 学习率
        self.w = np.zeros(train_x.shape[1])  # 权重
        self.b = 0  # 偏置
        self.support_vectors = None # 支持向量
        self.SV_x = None  # 支持向量的特征
        self.SV_y = None  # 支持向量的标签
        self.SV_on_boundary = None
        self.SV_on_boundary_x = None
        self.SV_on_boundary_y = None
    def fit(self):
        X = self.train_x * self.train_y  # 每个样本乘以其对应的标签
        # 应用增广拉格朗日算法求解alpha
        for i in range(self.max_iter):
            alpha_prev = np.copy(self.alpha)
            gradient = (np.dot(np.dot(X, X.T), self.alpha) - 1 + self.lamda * self.train_y + self.beta *
                        (np.dot(self.train_y.T, self.alpha) * self.train_y))

            self.alpha -= self.eta * gradient

            for j in range(self.train_num):
                if self.alpha[j] < 0:
                    self.alpha[j] = 0
                elif self.alpha[j] > self.C:
                    self.alpha[j] = self.C

            self.lamda += self.beta * (np.dot(self.train_y.T, self.alpha))

           #检查是否收敛
            if np.linalg.norm(self.alpha - alpha_prev) < 1e-5:
                break

            print("已迭代", i+1, "次")


        self.support_vectors = np.where(self.alpha > 0)[0]
        self.SV_x = self.train_x[self.support_vectors]
        self.SV_y = self.train_y[self.support_vectors]
        self.SV_on_boundary = np.where(np.logical_and(self.alpha> 0,self.alpha<self.C))[0]
        self.SV_on_boundary_x = self.train_x[self.SV_on_boundary]
        self.SV_on_boundary_y = self.train_y[self.SV_on_boundary]

        print("Training completed.")

    def calculate_w_b(self):
        self.w = np.sum(self.alpha[self.support_vectors] * self.train_y[self.support_vectors] * self.train_x[self.support_vectors], axis=0)
        self.b = np.mean(self.train_y[self.support_vectors] - np.dot(self.train_x[self.support_vectors], self.w))
        return self.w, self.b

def main():

    starttime = datetime.datetime.now()

    n = 100
    center1 = [1, 1]
    center2 = [6, 6]
    # 若改为线性不可分数据集，可以将center2改为[3,3]
    train_X = np.zeros((2 * n, 2))
    train_Y = np.zeros((2 * n, 1))

    for i in range(n):
        train_X[i] = np.random.normal(center1, 1, 2)
    for i in range(n, 2 * n):
        train_X[i] = np.random.normal(center2, 1, 2)

    train_Y[:n] = 1
    train_Y[n:] = -1
    # 画出各个数据点
    plt.figure(1, figsize=(7, 6))
    plt.plot(train_X[:n, 0], train_X[:n, 1], 'bo', markerfacecolor='none',linestyle='',markersize=7, label='Class 1')
    plt.plot(train_X[n:, 0], train_X[n:, 1], c='green',marker='*',linestyle='', markersize=7, label='Class 2')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.legend()

    svm = SupportVectorMachine(train_X, train_Y, 2 * n)
    svm.fit()
    w, b = svm.calculate_w_b()
    totalcount = 0
    for i in range(n):
        if w[0]*train_X[i,0]+w[1]*train_X[i,1]+b > 0:
            totalcount+=1
    for i in range(n,2*n):
        if w[0]*train_X[i,0]+w[1]*train_X[i,1]+b < 0:
            totalcount+=1
    accuracy = totalcount/(2*n)

    # 画出各个数据点
    plt.figure(2, figsize=(7, 6))
    plt.plot(train_X[:n, 0], train_X[:n, 1], 'bo', markerfacecolor='none',linestyle='',markersize=7, label='Class 1')
    plt.plot(train_X[n:, 0], train_X[n:, 1], c='green',marker='*',linestyle='', markersize=7, label='Class 2')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.legend()

    # 超平面S

    x = np.arange(-1, 7, 0.1)
    y = (w[0] * x + b) / -w[1]
    # 画出间隔边界
    x2 = np.arange(-1, 7, 0.1)
    y2 = (w[0] * x2 + b + 1) / -w[1]
    x3 = np.arange(-1, 7, 0.1)
    y3 = (w[0] * x3 + b - 1) / -w[1]
    plt.plot(x, y, label='Decision Boundary')
    plt.plot(x2,y2, 'k--',label='Margin Boundary')
    plt.plot(x3,y3, 'k--',label='Margin Boundary')
    #画出支持向量
    plt.plot(svm.SV_x[:, 0], svm.SV_x[:, 1],color='red', marker='s', markerfacecolor='none',linestyle='',
             markersize=7, label='Support Vectors')
    plt.plot(svm.SV_on_boundary_x[:, 0], svm.SV_on_boundary_x[:, 1],color='red', marker='s', linestyle='',
             markersize=7,label='Support Vectors on Boundary')

    stoptime = datetime.datetime.now()
    timedifference = stoptime - starttime
    print("Time:", timedifference.total_seconds())
    print("accuracy",accuracy)
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()