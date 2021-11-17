# PCA-SVM-
本模型使用了PCA、SVM、K折交叉验证方法对AR人脸数据集中的3120张图像进行训练和预测，并做了多重对比实验，进行了可视化分析。
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
