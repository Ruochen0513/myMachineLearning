import numpy as np
from matplotlib import pyplot as plt

class KNN:
    def __init__(self,k,train_X,train_Y):
        self.k = k
        self.train_X = train_X
        self.train_Y = train_Y

    def euclidean_distance(self,test_x,train_x):
        return np.sqrt(np.sum((test_x - train_x)**2))

    def KNN(self, test_x):
        # 计算x与所有训练样本的距离
        distances = [self.euclidean_distance(test_x, train_x) for train_x in self.train_X]
        # 根据距离排序，获取最近的k个样本的索引
        k_indices = np.argsort(distances)[:self.k]
        # 获取最近的k个样本的标签
        k_nearest_labels = [self.train_Y[i] for i in k_indices]
        # 投票选择最常见的标签作为预测结果
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

    def predict(self, test_X):
        Y_pred = []
        for test_x in test_X:
            most_common = self.KNN(test_x)
            Y_pred.append(most_common)
        return np.array(Y_pred)

def main():
    # 生成训练数据
    n = 2000
    X = np.random.rand(n, 2) * 10
    Y = np.zeros(n)

    for i in range(n):
        if 0 < X[i, 0] < 3 and 0 < X[i, 1] < 3:
            Y[i] = 1
        elif 0 < X[i, 0] < 3 and 3.5 < X[i, 1] < 6.5:
            Y[i] = 2
        elif 0 < X[i, 0] < 3 and 7 < X[i, 1] < 10:
            Y[i] = 3
        elif 3.5 < X[i, 0] < 6.5 and 0 < X[i, 1] < 3:
            Y[i] = 4
        elif 3.5 < X[i, 0] < 6.5 and 3.5 < X[i, 1] < 6.5:
            Y[i] = 5
        elif 3.5 < X[i, 0] < 6.5 and 7 < X[i, 1] < 10:
            Y[i] = 6
        elif 7 < X[i, 0] < 10 and 0 < X[i, 1] < 3:
            Y[i] = 7
        elif 7 < X[i, 0] < 10 and 3.5 < X[i, 1] < 6.5:
            Y[i] = 8
        elif 7 < X[i, 0] < 10 and 7 < X[i, 1] < 10:
            Y[i] = 9

    X = X[Y > 0]
    Y = Y[Y > 0]
    n = len(Y)

    # 生成测试数据
    m = 100
    Xt = np.random.rand(m, 2) * 10
    Yt = np.zeros(m)

    for i in range(m):
        if 0 < Xt[i, 0] < 3 and 0 < Xt[i, 1] < 3:
            Yt[i] = 1
        elif 0 < Xt[i, 0] < 3 and 3.5 < Xt[i, 1] < 6.5:
            Yt[i] = 2
        elif 0 < Xt[i, 0] < 3 and 7 < Xt[i, 1] < 10:
            Yt[i] = 3
        elif 3.5 < Xt[i, 0] < 6.5 and 0 < Xt[i, 1] < 3:
            Yt[i] = 4
        elif 3.5 < Xt[i, 0] < 6.5 and 3.5 < Xt[i, 1] < 6.5:
            Yt[i] = 5
        elif 3.5 < Xt[i, 0] < 6.5 and 7 < Xt[i, 1] < 10:
            Yt[i] = 6
        elif 7 < Xt[i, 0] < 10 and 0 < Xt[i, 1] < 3:
            Yt[i] = 7
        elif 7 < Xt[i, 0] < 10 and 3.5 < Xt[i, 1] < 6.5:
            Yt[i] = 8
        elif 7 < Xt[i, 0] < 10 and 7 < Xt[i, 1] < 10:
            Yt[i] = 9

    Xt = Xt[Yt > 0]
    Yt = Yt[Yt > 0]
    m = len(Yt)

    # 创建KNN对象并进行预测
    knn = KNN(k=5, train_X=X, train_Y=Y)
    Y_pred = knn.predict(Xt)

    # 可视化测试数据和预测结果
    plt.figure(figsize=(7, 6))

    plt.plot(X[Y == 1, 0], X[Y == 1, 1], color = 'red',marker='o', linestyle='', markersize=3,
             label='Class ' + '1')
    plt.plot(X[Y == 2, 0], X[Y == 2, 1], color = 'orange',marker='o', linestyle='', markersize=3,
             label='Class ' + '2')
    plt.plot(X[Y == 3, 0], X[Y == 3, 1], color = 'blue',marker='o', linestyle='', markersize=3,
             label='Class ' + '3')
    plt.plot(X[Y == 4, 0], X[Y == 4, 1], color = 'yellow',marker='*', linestyle='', markersize=3,
             label='Class ' + '4')
    plt.plot(X[Y == 5, 0], X[Y == 5, 1], color = 'green',marker='*', linestyle='', markersize=3,
             label='Class ' + '5')
    plt.plot(X[Y == 6, 0], X[Y == 6, 1], color = 'purple',marker='*', linestyle='', markersize=3,
             label='Class ' + '6')
    plt.plot(X[Y == 7, 0], X[Y == 7, 1], color = 'brown',marker='+', linestyle='', markersize=3,
             label='Class ' + '7')
    plt.plot(X[Y == 8, 0], X[Y == 8, 1], color = 'black',marker='+', linestyle='', markersize=3,
             label='Class ' + '8')
    plt.plot(X[Y == 9, 0], X[Y == 9, 1], color = 'pink',marker='+', linestyle='', markersize=3,
             label='Class ' + '9')

    plt.plot(Xt[Y_pred == 1, 0], Xt[Y_pred == 1, 1], color = 'red',marker='s', linestyle='', markersize=7,
             label='Predicted Class ' + '1')
    plt.plot(Xt[Y_pred == 2, 0], Xt[Y_pred == 2, 1], color = 'orange',marker='s', linestyle='', markersize=7,
             label='Predicted Class ' + '2')
    plt.plot(Xt[Y_pred == 3, 0], Xt[Y_pred == 3, 1], color = 'blue',marker='s', linestyle='', markersize=7,
             label='Predicted Class ' + '3')
    plt.plot(Xt[Y_pred == 4, 0], Xt[Y_pred == 4, 1], color = 'yellow',marker='s', linestyle='', markersize=7,
             label='Predicted Class ' + '4')
    plt.plot(Xt[Y_pred == 5, 0], Xt[Y_pred == 5, 1], color = 'green',marker='s', linestyle='', markersize=7,
             label='Predicted Class ' + '5')
    plt.plot(Xt[Y_pred == 6, 0], Xt[Y_pred == 6, 1],color = 'purple', marker='s', linestyle='', markersize=7,
             label='Predicted Class ' + '6')
    plt.plot(Xt[Y_pred == 7, 0], Xt[Y_pred == 7, 1],color = 'brown', marker='s', linestyle='', markersize=7,
             label='Predicted Class ' + '7')
    plt.plot(Xt[Y_pred == 8, 0], Xt[Y_pred == 8, 1], color = 'black',marker='s', linestyle='', markersize=7,
             label='Predicted Class ' + '8')
    plt.plot(Xt[Y_pred == 9, 0], Xt[Y_pred == 9, 1], color = 'pink',marker='s', linestyle='', markersize=7,
             label='Predicted Class ' + '9')

    plt.xlabel('x axis')
    plt.ylabel('y axis')

    error_rate = np.sum(Y_pred != Yt) / m
    error_rate_text = "Error Rate:{:.2f}%".format(error_rate * 100)
    plt.text(0, -1.5, error_rate_text, fontsize=12, ha='center')
    plt.show()
    print(f"错误率为：{error_rate}")
if __name__ == "__main__":
    main()