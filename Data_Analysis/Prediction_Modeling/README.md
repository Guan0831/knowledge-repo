# 🔮 预测与建模 (Prediction & Modeling)

本目录专注于利用历史数据构建各种预测模型，从而**预测**未来的结果、进行**分类**判断、或者识别复杂的模式。这里涵盖了丰富的**机器学习 (Machine Learning, ML)** 和**深度学习 (Deep Learning, DL)** 相关概念、算法和实践示例。

无论你是想预测房价、识别图像中的物体、还是对用户行为进行分类，你都能在这里找到相应的工具和方法。

## 核心主题与算法

我们按照模型类型和应用场景组织了内容：

-   ### 监督学习 (Supervised Learning)
    这类模型从带有标签（即已知输出）的数据中学习，然后对新数据进行预测。

    -   **回归模型 (Regression Models):** 用于预测连续数值型输出。
        -   [线性回归 (Linear Regression)](<./01-Regression-Models/Linear-Regression.md>)
        -   [多项式回归 (Polynomial Regression)](<./01-Regression-Models/Polynomial-Regression.md>)
        -   [决策树回归 (Decision Tree Regressor)](<./01-Regression-Models/Decision-Tree-Regression.md>)
        -   [支持向量回归 (Support Vector Regression - SVR)](<./01-Regression-Models/SVR.md>)
        -   ... (更多回归模型，如岭回归、Lasso 回归、K-近邻回归等)

    -   **分类模型 (Classification Models):** 用于预测离散类别型输出。
        -   [逻辑回归 (Logistic Regression)](<./02-Classification-Models/Logistic-Regression.md>)
        -   [支持向量机 (Support Vector Machines - SVM)](<./02-Classification-Models/SVM.md>)
        -   [决策树分类 (Decision Tree Classifier)](<./02-Classification-Models/Decision-Tree-Classification.md>)
        -   [K-近邻算法 (K-Nearest Neighbors - KNN)](<./02-Classification-Models/KNN.md>)
        -   [朴素贝叶斯 (Naive Bayes)](<./02-Classification-Models/Naive-Bayes.md>)
        -   ... (更多分类模型，如感知机、神经网络分类器等)

-   ### 集成学习 (Ensemble Learning)
    这类方法通过结合多个学习器来提高预测性能和稳定性。

    -   [随机森林 (Random Forest)](<./03-Ensemble-Methods/Random-Forest.md>)
    -   [梯度提升 (Gradient Boosting - GBDT)](<./03-Ensemble-Methods/Gradient-Boosting.md>)
    -   [XGBoost / LightGBM](<./03-Ensemble-Methods/XGBoost-LightGBM.md>)
    -   ... (更多集成方法，如 AdaBoost、Stacking 等)

-   ### 深度学习基础 (Deep Learning Basics)
    基于神经网络的机器学习子领域，在图像、语音和自然语言处理等领域表现卓越。

    -   [前馈神经网络 (Feedforward Neural Networks - FNN)](<./04-Deep-Learning-Basics/Feedforward-NN.md>)
    -   [卷积神经网络 (Convolutional Neural Networks - CNN)](<./04-Deep-Learning-Basics/CNN.md>)
    -   [循环神经网络 (Recurrent Neural Networks - RNN)](<./04-Deep-Learning-Basics/RNN.md>)
    -   [深度学习框架 (TensorFlow/PyTorch) 简介](<./04-Deep-Learning-Basics/DL-Frameworks-Intro.md>)
    -   ... (更多深度学习概念，如激活函数、损失函数、优化器、预训练模型等)

-   ### 模型选择与调优 (Model Selection & Tuning)
    确保模型具有良好的泛化能力，并找到最优的模型参数。

    -   [交叉验证 (Cross-Validation)](<./05-Model-Selection-Tuning/Cross-Validation.md>)
    -   [超参数调优 (Hyperparameter Tuning)](<./05-Model-Selection-Tuning/Hyperparameter-Tuning.md>)
    -   [正则化 (Regularization)](<./05-Model-Selection-Tuning/Regularization.md>)
    -   [特征选择与工程 (Feature Selection & Engineering)](<./05-Model-Selection-Tuning/Feature-Selection-Engineering.md>)
    -   ... (更多模型优化技术)
---

[🔙 返回数据分析主目录](<../README.md>)
[回到知识体系库根目录](<../../../README.md>)
