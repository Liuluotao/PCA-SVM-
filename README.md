# PCA-SVM-
本项目做了两部分工作，一个是尝试手动实现PCA，一步步推导（在代码PCA中）；第二个是使用Sklearn里面的包PCA和SVM，直接实现（在代码main里面）。
手动实现的程序使用了使用了数据库CASIA-FaceV5中的100个不同亚洲人的每人五张图像，由于准确率较低，故选择使用Sklearn里面的包。第二种使用PCA、SVM、K折交叉验证方法对AR人脸数据集中的3120张图像进行训练和预测，并做了多重对比实验，进行了可视化分析，得到了不错的结果。
## 开发环境
* Python 3.8 x64
## IDE
PyCharm 2021.1.3 x64
## 依赖包
* numpy
* scipy
* scikit-learn
* PIL
## 代码内容
１.PCA保留的主成分数n_components对准确率的影响
２.准确率随不同gamma和核函数变化曲线
３. k重交叉验证的k值对准确率的影响
