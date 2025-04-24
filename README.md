your-knowledge-repo/
#这里是一个简单的暂定框架，从这框架出发一点点完善x
├── .gitignore
├── LICENSE
├── README.md          # 更新此文件，加入新的顶级分类和更详细的导航
│
├── Algorithms/        # 新增顶级分类：算法与数据结构 (适合 LeetCode)
│   ├── README.md      # 介绍本分类，包含 LeetCode 链接等
│   ├── Data-Structures/ # 数据结构 (数组、链表、树、图等)
│   │   ├── README.md
│   │   ├── arrays.md
│   │   ├── linked-lists.md
│   │   └── ...
│   ├── Common-Algorithms/ # 常见算法 (排序、搜索、动态规划、贪心等)
│   │   ├── README.md
│   │   ├── sorting.md
│   │   ├── searching.md
│   │   └── ...
│   ├── LeetCode-Solutions/ # LeetCode 题目解答
│   │   ├── README.md       # 说明解答的组织方式 (按题号、按难度、按主题)
│   │   ├── easy/           # 简单题
│   │   │   ├── 0001-Two-Sum.py # Python 代码解答
│   │   │   ├── 0001-Two-Sum.md # 可能包含思路分析或多种解法对比
│   │   │   └── ...
│   │   ├── medium/         # 中等题
│   │   └── hard/           # 困难题
│   └── ...
│
├── Data-Analysis/
│   ├── README.md      # 更新导航
│   ├── 01-Understand-Explore/
│   │   ├── README.md
│   │   ├── loading-data.md
│   │   ├── descriptive-stats.md
│   │   ├── initial-viz.md
│   │   ├── python-data-loading-snippet.py # Python 代码示例
│   │   └── ...
│   ├── 02-Clean-Preprocess/
│   │   ├── README.md
│   │   ├── handle-missing-values.md
│   │   ├── handle-outliers.md
│   │   ├── python-data-cleaning-code.py # Python 代码示例
│   │   └── ...
│   ├── 03-Patterns-Relationships/ ...
│   ├── 04-Prediction-Modeling/ # 目的：预测未来趋势与建模 (重点加入 DeepML/ML)
│   │   ├── README.md
│   │   ├── regression-basics.md
│   │   ├── classification-models.md
│   │   ├── model-evaluation.md
│   │   ├── deep-learning-intro.md
│   │   ├── CNN-Example/         # CNN 实例可以是一个子目录
│   │   │   ├── README.md        # CNN 实例说明
│   │   │   ├── cnn_mnist_example.ipynb # Jupyter Notebook 实例
│   │   │   ├── cnn_model.py       # 模型代码
│   │   │   └── ...
│   │   ├── RNN-LSTM-Example/    # RNN/LSTM 实例
│   │   │   ├── README.md
│   │   │   ├── lstm_time_series.ipynb
│   │   │   └── ...
│   │   ├── Traditional-ML-Examples/ # 传统机器学习实例
│   │   │   ├── README.md
│   │   │   ├── linear-regression-sklearn.ipynb
│   │   │   └── ...
│   │   └── ...
│   ├── 05-Evaluate-Compare/ ...
│   ├── 06-Storytelling-Communicate/ ...
│   └── Kaggle-Projects/       # 新增子目录：Kaggle 项目示例
│       ├── README.md          # 介绍此处的 Kaggle 项目集合
│       ├── Titanic-Survival/  # 项目名称作为一个文件夹
│       │   ├── README.md      # 项目描述、目标、所用技术
│       │   ├── notebooks/     # 存放 Jupyter Notebook (.ipynb) 文件
│       │   │   ├── 01-EDA.ipynb
│       │   │   ├── 02-Feature-Engineering.ipynb
│       │   │   ├── 03-Model-Training.ipynb
│       │   │   └── ...
│       │   ├── src/           # 存放 Python 脚本 (.py)
│       │   ├── data/          # 存放数据集 (注意文件大小和隐私)
│       │   └── ...
│       ├── House-Price-Prediction/ # 另一个项目
│       │   └── ...
│       └── ...
│
├── Data-Processing/
│   ├── README.md      # 更新导航
│   ├── 01-Acquire-Ingest/
│   │   ├── README.md
│   │   ├── web-scraping-basics.md
│   │   ├── python-scraping-code.py # Python 代码示例
│   │   └── ...
│   ├── 02-Storage-Management/ ...
│   ├── 03-Clean-Transform/
│   │   ├── README.md
│   │   ├── etl-elt-concepts.md
│   │   ├── data-type-conversion.md
│   │   ├── python-data-transformation-script.py # Python 脚本示例
│   │   └── ...
│   ├── 04-Pipelines-Automation/ ...
│   │   ├── README.md
│   │   ├── workflow-scheduling.md
│   │   ├── simple-pipeline-example.py # 简单管道代码示例
│   │   └── ...
│   ├── 05-Big-Data-Processing/ ...
│   └── 06-Quality-Governance/ ...
│
├── Mechanical-Engineering/
│   ├── README.md      # 更新导航
│   ├── 01-Conceptual-Design/ ...
│   ├── 02-Part-Structure-Design/
│   │   ├── README.md
│   │   ├── stress-strain-basics.md
│   │   ├── fasteners-guide.md
│   │   ├── tolerancing-basics.md
│   │   ├── python-fea-script-concept.py # Python 在结构分析中的概念性代码
│   │   └── ...
│   ├── 03-Mechanisms-Dynamics/
│   │   ├── README.md
│   │   ├── kinematics-intro.md
│   │   ├── vibration-analysis.md
│   │   ├── python-simulation-code.py # Python 模拟代码示例
│   │   └── ...
│   ├── 04-Thermo-Fluid-Systems/ ...
│   │   ├── README.md
│   │   ├── heat-transfer-modes.md
│   │   ├── pump-selection.md
│   │   ├── python-cfd-postprocessing.py # Python 用于 CFD 后处理示例
│   │   └── ...
│   ├── 05-Manufacturing-Planning/ ...
│   ├── 06-System-Integration-Testing/
│   │   ├── README.md
│   │   ├── test-plan-design.md
│   │   ├── python-data-acquisition-code.py # Python 用于测试数据采集
│   │   └── ...
│   ├── 07-Operations-Maintenance/ ...
│   └── 08-Technical-Communication/ ...
│
└── Cross-Cutting-Concepts/ # 跨领域通用概念
    ├── README.md
    ├── problem-solving-methods.md
    ├── project-management-basics.md
    ├── python-basic-syntax.md # 非常基础且通用的 Python 语法可以放这里
    └── ...
