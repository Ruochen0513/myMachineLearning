import numpy as np
from matplotlib import pyplot as plt

class Perceptron:

    def __init__(self, train_x, train_y,test_x,test_y, eta=1):
        # 超参数：步长
        self.eta = eta
        # 初始化参数w, b
        self.w = np.zeros(train_x.shape[1])
        self.b = 0
        # 保存训练集
        self.train_x = train_x
        self.train_y = train_y
        # 保存测试集
        self.test_x = test_x
        self.test_y = test_y
        self.n_test = 10
        # 迭代次数
        self.iteration_num = 1000

    def train(self):

        #具有最小总损失的参数
        best_w = self.w.copy()
        best_b = self.b
        min_total_loss = float('inf')

        count =0
        while self.iteration_num > 0:
            # 计算误分类点到超平面的距离
            distances = self.train_y * (np.dot(self.train_x, self.w.T) + self.b)
            misclassified_indices = np.where(distances <= 0)[0]         #所有误分类点的集合

            misclassified_distances = -distances[misclassified_indices] #误分类点到超平面的距离
            # 计算损失函数（距离之和）
            total_loss = np.sum(misclassified_distances)
            # 如果此时的损失函数最小，就记录下这次的w，b参数
            if total_loss < min_total_loss and total_loss > 0:
                best_b = self.b
                best_w = self.w.copy()
                min_total_loss = total_loss

            # 打乱顺序，随机取出一个误分类点进行参数更新
            np.random.shuffle(misclassified_indices)
            x = self.train_x[misclassified_indices[0]]
            y = self.train_y[misclassified_indices[0]]

            self.w += self.eta * y * x
            self.b += self.eta * y
            # 更新迭代次数
            self.iteration_num -= 1
            count +=1
            print("已迭代",count,"次")

        self.w = best_w.copy()
        self.b = best_b
        print("Accuracy: {:.2f}%".format(self.accuracy() * 100))
        return self.w, self.b


    # 预测函数
    def predict(self,x):

        if np.dot(x, self.w.T) + self.b > 0:
            return 1
        else:
            return -1

    def accuracy(self):

        correct_count = 0
        for i in range(self.n_test *2):
            prediction = self.predict(self.test_x[i])
            if prediction == self.test_y[i]:
               correct_count += 1
        accuracy = correct_count / (self.n_test*2)
        return accuracy

    # 绘制分类结果
    def draw(self):
        # 训练集点，正例蓝色，负例红色
        x1 = self.train_x[np.where(self.train_y > 0)][:, 0]
        y1 = self.train_x[np.where(self.train_y > 0)][:, 1]
        x2 = self.train_x[np.where(self.train_y < 0)][:, 0]
        y2 = self.train_x[np.where(self.train_y < 0)][:, 1]
        # 测试集点，正负例绿色
        x4 = self.test_x[np.where(self.test_y > 0)][:, 0]
        y4 = self.test_x[np.where(self.test_y > 0)][:, 1]
        x5 = self.test_x[np.where(self.test_y < 0)][:, 0]
        y5 = self.test_x[np.where(self.test_y < 0)][:, 1]
        # 超平面S
        x3 = np.arange(-2, 6, 0.1)
        y3 = (self.w[0] * x3 + self.b) / -self.w[1]

        plt.plot(x1, y1, 'b*', label='TrainClass +1',markersize = 7)
        plt.plot(x2, y2, 'ro', label='TrainClass -1',markersize = 7)
        plt.plot(x4, y4, 'g*', label='testClass +1', markersize=5)
        plt.plot(x5, y5, 'go', label='testClass -1', markersize=5)
        plt.plot(x3, y3, label='Decision Boundary')
        # 添加准确率文本
        acc_text = "Accuracy: {:.2f}%".format(self.accuracy() * 100)
        plt.text(-1, -1, acc_text, fontsize=12, ha='center')
        plt.title('Perceptron Classification')
        plt.legend()
        plt.show()

def main():
    # 训练数据集
    n = 100         #样本量大小
    center1 = [1,1]  #第一类数据中心
    center2 = [3,4]  #第二类数据中心
    train_X = np.zeros((2*n,2))  #训练数据坐标
    train_Y = np.zeros(2*n)  #训练数据标签

    for i in range(0,n):
        train_X[i] = np.random.normal(center1,1,2)
    for i in range(n,2*n):
        train_X[i] = np.random.normal(center2,1,2)

    for i in range(0,n):
        train_Y[i] = -1
    for i in range(n, 2*n):
        train_Y[i] = 1

    # 测试数据集
    n_test = 10  # 样本量大小
    test_X = np.zeros((2 * n_test, 2))  # 训练数据坐标
    test_Y = np.zeros(2 *n_test)  # 训练数据标签

    for i in range(0, n_test):
        test_X[i] = np.random.normal(center1, 1, 2)
    for i in range(n_test, 2 * n_test):
        test_X[i] = np.random.normal(center2, 1, 2)

    for i in range(0, n_test):
        test_Y[i] = -1
    for i in range(n_test, 2 * n_test):
        test_Y[i] = 1


    perceptron = Perceptron(train_X, train_Y,test_X,test_Y)
    perceptron.train()
    perceptron.draw()



if __name__ == "__main__":
    main()
