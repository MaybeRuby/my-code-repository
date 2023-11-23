单独解释一下
```
model = xgb.XGBClassifier(objective='binary:logistic', colsample_bytree=0.7, learning_rate=0.1,
                max_depth=5, alpha=1, n_estimators=15)
```
* 列采样比例 (colsample_bytree)、学习率 (learning_rate)、树的最大深度 (max_depth)、L1 正则化参数 (alpha) 和树的数量 (n_estimators)。
* objective(string or callable)：在回归问题一般使用reg:squarederror ，即MSE均方误差。二分类问题一般使用binary:logistic, 多分类问题一般使用multi:softmax。
* eval_metric (str, list of str, or callable, optional, default=None):指定模型评估指标。常见的值包括：
  * 'rmse'：均方根误差（用于回归问题）。
  * 'mae'：平均绝对误差（用于回归问题）。
  * 'logloss'：对数损失（用于二分类问题）。
  * 'error'：错误率（用于分类问题）。
  * 自定义评估指标。
* base_score (float, optional, default=0.5):在计算初始的预测分数时使用的基准分数。
* booster (string, optional, default='gbtree'):同框架参数，指定使用的弱提升器类型。可以是默认的gbtree, 也就是CART决策树，还可以是线性弱学习器gblinear以及DART。一般来说，我们使用gbtree就可以了，不需要调参。
* learning_rate (float, optional, default=0.3):学习率，控制每次提升的步长。较小的学习率通常需要较大的 n_estimators。
* n_estimators：是非常重要的要调的参数，它关系到我们XGBoost模型的复杂度，因为它代表了我们决策树弱学习器的个数。这个参数对应sklearn GBDT的n_estimators。n_estimators太小，容易欠拟合，n_estimators太大，模型会过于复杂，一般需要调参选择一个适中的数值。
* gamma (float, optional, default=0):指定节点分裂时，损失函数的降低阈值。节点分裂只有在分裂后的损失函数值低于阈值时才会进行。
* max_depth (int, optional, default=6):每棵树的最大深度。较大的深度可能会导致过拟合。
* min_child_weight (float, optional, default=1):叶子节点的最小样本权重和。用于控制过拟合，较大的值使算法更保守。
* subsample (float, optional, default=1):训练每棵树时使用的样本比例。可以避免过拟合。
* colsample_bytree (float, optional, default=1):每棵树的特征（列）采样比例。避免模型对于某些特征过于依赖。
* lambda (float, optional, default=1):L2 正则化权重。用于控制模型复杂度。
* alpha (float, optional, default=0):L1 正则化权重。用于控制模型复杂度。
* scale_pos_weight (float, optional, default=1):在类别不平衡问题中，控制正负权重的平衡。
* n_jobs (int, optional, default=-1):并行线程数，用于拟合和预测。默认值 -1 表示使用所有可用的 CPU 核心。

```
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pylab as plt
%matplotlib inline

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

selected_features = ['Pclass', 'Sex', 'Age','SibSp']
df_selected = train_df[selected_features + ['Survived']]
df_test = test_df[selected_features]
df_selected = df_selected.dropna()

X = df_selected.drop('Survived', axis=1)
y = df_selected['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

model = xgb.XGBClassifier(objective='binary:logistic', colsample_bytree=0.7, learning_rate=0.1,
                max_depth=5, alpha=1, n_estimators=15)

model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```
