import os
import operator
from numpy import *
import matplotlib.pyplot as plt
import cv2


# define PCA
def pca(data, p):
    data = float32(mat(data))  # mat 变成矩阵格式
    rows, cols = data.shape  # 取大小
    data_mean = mean(data, 0)  # 对列求均值  每个特征维度的平均值
    data_mean_all = tile(data_mean, (rows, 1))  # 复制出row行，1列(保留data_mean列）
    Z = data - data_mean_all
    Z = Z.T  # d*m
    T1 = Z * Z.T  # d*d
    D, V = linalg.eig(T1)  # 特征值与特征向量
    # V = V.T  # 得到的特征向量是一行一行的 ， 转变成一列一列的
    # D = D.tolist()  # 为什么多了排在第一的0
    # del D[0]
    sorted_Index = argsort(-D)  # 将元素从小到大排列，取出对应的索引
    # 为什么 0 在最开始，32424 在最后
    V1 = V[:, sorted_Index[0:p]]
    for i in range(p):
        L = linalg.norm(V1[:, i])  # 求二范数
        V1[:, i] = V1[:, i] / L   # 特征向量归一化
    V1 = V1.T

    data_new = V1 * Z  # 降维后的有方向向量（投影后的样本）  V1 k*d  Z d*m  新数据维度 k*m
    data_new = data_new.T
    return data_new, data_mean, V1   # 现在V1为投影后的向量长度吧？不是投影矩阵？


# covert image to vector
def img2vector(filename):
    img = cv2.imread(filename, 0)  # read as 'gray'  读取图像
    rows, cols = img.shape
    # imgVector = zeros((1, rows * cols))  # create a none vectore:to raise speed
    imgVector = reshape(img, (1, rows * cols))  # change img from 2D to 1D
    return imgVector


# load dataSet
def loadDataSet(k):  # choose k(0-5) people as traintest for everyone
    #  step 1:Getting data set
    print("--Getting data set---")
    # note to use '/'  not '\'
    # dataSetDir = 'D:/A-LLT/Course/Machine_Learning/Data'
    dataSetDir = './Data'
    # choose = random.permutation(5) + 1  # 随机排序1-5 (0-9）+1   有何用？
    train_face = zeros((100 * k, 32424))  # 像素
    train_face_number = zeros(100 * k)
    test_face = zeros((100 * (5 - k), 32424))
    test_face_number = zeros(100 * (5 - k))    # 100个不同的人，每个人有5个图片，K张为训练集，5-k为测试集
    for i in range(100):  # 100 sample people
        people_num = '{0:03}'.format((i))
        for j in range(5):  # everyone has 10 different face
            num = '{0:03}'.format((i)) + '_' + str(j)
            if j < k:
                filename = dataSetDir + '/' + str(people_num) + '/' + str(num) + '.bmp'  # .pgm
                img = img2vector(filename)  # 返回一个行向量
                train_face[i * k + j, :] = img    # 注意i为从0-39，将第i个人的第j张图片放到对应位置
                train_face_number[i * k + j] = i  # 表示第几个样本人
            else:
                filename = dataSetDir + '/' + str(people_num) + '/' + str(num) + '.bmp'
                img = img2vector(filename)
                test_face[i * (5 - k) + (j - k), :] = img
                test_face_number[i * (5 - k) + (j - k)] = i

    return train_face, train_face_number, test_face, test_face_number


# calculate the accuracy of the test_face
def facefind():
    # Getting data set
    train_face, train_face_number, test_face, test_face_number = loadDataSet(3)
    # PCA training to train_face
    data_train_new, data_mean, V = pca(train_face, 100)  # 把每个脸的特征降到30维
    num_train = data_train_new.shape[0]
    num_test = test_face.shape[0]
    temp_face = test_face - tile(data_mean, (num_test, 1))  # 测试集数据的标准化也是减训练集的平均值

    data_test_new = V * temp_face.T  # 得到测试脸在特征向量下的数据 V是不是有问题？
    data_test_new = array(data_test_new)  # mat change to array
    data_test_new = data_test_new.T
    data_train_new = array(data_train_new)

    true_num = 0
    for i in range(num_test):
        testFace = data_test_new[i, :]
        diffMat = data_train_new - tile(testFace, (num_train, 1))
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis=1)  # 对每一行求和
        # 为什么这个没有变化
        sortedDistIndicies = sqDistances.argsort()  # 排序后的列表中的数在原列表的索引。
        indexMin = sortedDistIndicies[1]
        if train_face_number[indexMin] == test_face_number[i]:
            true_num += 1
    accuracy = float(true_num) / num_test
    print('The classify accuracy is: %.2f%%' % (accuracy * 100))


if __name__ == '__main__':
    facefind()


