import numpy as np
from matplotlib import pyplot as plt

class NaiveBayesClassifier:
    def __init__(self):
        # 初始化均值、方差和先验概率
        self.mean = {}
        self.var = {}
        self.priors = {}

    def fit(self, X, Y):
        # 获取类别数量
        self.classes = np.unique(Y)
        # 计算先验概率
        n_samples, n_features = X.shape
        for c in self.classes:
            # 选择属于类别 c 的数据
            X_c = X[Y == c]
            # 计算均值
            self.mean[c] = np.mean(X_c, axis=0)
            # 计算方差
            self.var[c] = np.var(X_c, axis=0)
            # 计算先验概率
            self.priors[c] = X_c.shape[0] / n_samples

    def likelihood(self, class_idx, x):
        # 计算正态分布的似然（对于每个特征）
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-(x - mean) ** 2 / 2 * var )
        denominator = np.sqrt(2 * np.pi * var )
        likelihood = numerator / denominator
        # 对于多维特征，需要计算所有特征的似然乘积
        return np.prod(likelihood)

    def posterior(self, x):
        # 计算后验概率（对数形式）
        posteriors = []
        for c in self.classes:
            prior = self.priors[c]
            conditional = self.likelihood(c, x)

            posterior = prior * conditional
            posteriors.append(posterior)
            # 返回概率最大的类别
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        # 对每个样本点进行预测
        y_pred = [self.posterior(x) for x in X]
        return np.array(y_pred)


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

    # 生成噪声
    nn = 2000
    for i in range(n, nn):
        x_new = np.random.rand(1, 2) * 10
        X = np.vstack((X, x_new))
        Y = np.append(Y, np.ceil(np.random.rand(1) * 9))
    print("噪声比为：", (nn - n) / nn)
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

    # 画出第一张图：训练数据、噪声和未预测的测试数据
    plt.figure(1, figsize=(7, 6))
    colors = ['red', 'orange', 'blue', 'yellow', 'green', 'gray', 'brown', 'black', 'pink']
    markers = ['o', 'o', 'o', '*', '*', '*', '+', '+', '+']
    for i in range(1, 10):
        plt.plot(X[Y == i, 0], X[Y == i, 1], color=colors[i - 1], marker=markers[i - 1], linestyle='', markersize=3,
                 label='Class ' + str(i))

    for i in range(m):
        plt.plot(Xt[i, 0], Xt[i, 1], color='purple', marker='s', linestyle='', markersize=7)
        # 所有未经预测的初始测试点都是紫色方形的点
    plt.xlabel('x axis')
    plt.ylabel('y axis')

    # 模型训练
    classifier = NaiveBayesClassifier()
    classifier.fit(X, Y)
    # 模型预测
    Y_pred = classifier.predict(Xt)

    # 画出第二张图：训练数据、噪声和已预测的测试数据
    plt.figure(2, figsize=(7, 6))
    for i in range(1, 10):
        plt.plot(X[Y == i, 0], X[Y == i, 1], color=colors[i - 1], marker=markers[i - 1], linestyle='', markersize=3,
                 label='Class ' + str(i))

    for i in range(1, 10):
        plt.plot(Xt[Y_pred == i, 0], Xt[Y_pred == i, 1], color=colors[i - 1], marker='s', linestyle='', markersize=7,
                 label='Predicted Class ' + str(i))

    error_rate = np.sum(Y_pred != Yt) / m
    error_rate_text = "Error Rate:{:.2f}%".format(error_rate * 100)
    plt.text(0, -1.5, error_rate_text, fontsize=12, ha='center')

    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.show()

    print(f"错误率为：{error_rate}")
if __name__ == '__main__':
    main()