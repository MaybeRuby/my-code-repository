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
# 列采样比例 (colsample_bytree)、学习率 (learning_rate)、树的最大深度 (max_depth)、
# L1 正则化参数 (alpha) 和树的数量 (n_estimators)。

model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
```
