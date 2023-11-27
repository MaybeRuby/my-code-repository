## 数据加载
该部分数据是基于拍拍贷比赛截取的一部分特征，随机选择了一部分数据做训练集，一部分做测试集。针对其中gender、cell_province等类别特征，直接进行重新编码处理。原始数据的lable是0-32，共有33个类别的数据。针对二分类任务，将原始label为32的数据直接转化为1，label为其他的数据转为0；回归问题就是将这些类别作为待预测的目标值。代码如下：其中gc是释放不必要的内存。

```python
## category feature one_hot
test_data['label'] = -1
data = pd.concat([train_data, test_data])
cate_feature = ['gender', 'cell_province', 'id_province', 'id_city', 'rate', 'term']
for item in cate_feature:
    data[item] = LabelEncoder().fit_transform(data[item])

train = data[data['label'] != -1]
test = data[data['label'] == -1]

## Clean up the memory
del data, train_data, test_data
gc.collect()

## get train feature
del_feature = ['auditing_date', 'due_date', 'label']
features = [i for i in train.columns if i not in del_feature]

## Convert the label to two categories
train['label'] = train['label'].apply(lambda x: 1 if x==32 else 0)
train_x = train[features]
train_y = train['label'].values
test = test[features]
```

## 二分类任务

```python
params = {'num_leaves': 60, #结果对最终效果影响较大，越大值越好，太大会出现过拟合
          'min_data_in_leaf': 30,
          'objective': 'binary', #定义的目标函数
          'max_depth': -1,
          'learning_rate': 0.03,
          "min_sum_hessian_in_leaf": 6,
          "boosting": "gbdt",
          "feature_fraction": 0.9,  #提取的特征比率
          "bagging_freq": 1,
          "bagging_fraction": 0.8,
          "bagging_seed": 11,
          "lambda_l1": 0.1,             #l1正则
          # 'lambda_l2': 0.001,     #l2正则
          "verbosity": -1,
          "nthread": -1,                #线程数量，-1表示全部线程，线程越多，运行的速度越快
          'metric': {'binary_logloss', 'auc'},  ##评价函数选择
          "random_state": 2019, #随机数种子，可以防止每次运行的结果不一致
          # 'device': 'gpu' ##如果安装的事gpu版本的lightgbm,可以加快运算
          }

folds = KFold(n_splits=5, shuffle=True, random_state=2019)
prob_oof = np.zeros((train_x.shape[0], ))
test_pred_prob = np.zeros((test.shape[0], ))


## train and predict
feature_importance_df = pd.DataFrame()
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train)):
    print("fold {}".format(fold_ + 1))
    trn_data = lgb.Dataset(train_x.iloc[trn_idx], label=train_y[trn_idx])
    val_data = lgb.Dataset(train_x.iloc[val_idx], label=train_y[val_idx])


    clf = lgb.train(params,
                    trn_data,
                    num_round,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=20,
                    early_stopping_rounds=60)
    prob_oof[val_idx] = clf.predict(train_x.iloc[val_idx], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    test_pred_prob += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits

threshold = 0.5
for pred in test_pred_prob:
    result = 1 if pred > threshold else 0
```
上面的参数中目标函数采用的事binary，评价函数采用的是{'binary_logloss', 'auc'}，可以根据需要对评价函数做调整，可以设定一个或者多个评价函数；'num_leaves'对最终的结果影响较大，如果值设置的过大会出现过拟合现象。

针对模型训练部分，采用的事5折交叉训练的方法，常用的5折统计有两种：StratifiedKFold和KFold，其中最大的不同是StratifiedKFold分层采样交叉切分，确保训练集，测试集中各类别样本的比例与原始数据集中相同，实际使用中可以根据具体的数据分别测试两者的表现。

最后fold_importance_df表存放的事模型的特征重要性，可以方便分析特征重要性

## 多分类任务

```python
params = {'num_leaves': 60,
          'min_data_in_leaf': 30,
          'objective': 'multiclass',
          'num_class': 33,
          'max_depth': -1,
          'learning_rate': 0.03,
          "min_sum_hessian_in_leaf": 6,
          "boosting": "gbdt",
          "feature_fraction": 0.9,
          "bagging_freq": 1,
          "bagging_fraction": 0.8,
          "bagging_seed": 11,
          "lambda_l1": 0.1,
          "verbosity": -1,
          "nthread": 15,
          'metric': 'multi_logloss',
          "random_state": 2019,
          # 'device': 'gpu' 
          }


folds = KFold(n_splits=5, shuffle=True, random_state=2019)
prob_oof = np.zeros((train_x.shape[0], 33))
test_pred_prob = np.zeros((test.shape[0], 33))

## train and predict
feature_importance_df = pd.DataFrame()
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train)):
    print("fold {}".format(fold_ + 1))
    trn_data = lgb.Dataset(train_x.iloc[trn_idx], label=train_y.iloc[trn_idx])
    val_data = lgb.Dataset(train_x.iloc[val_idx], label=train_y.iloc[val_idx])

    clf = lgb.train(params,
                    trn_data,
                    num_round,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=20,
                    early_stopping_rounds=60)
    prob_oof[val_idx] = clf.predict(train_x.iloc[val_idx], num_iteration=clf.best_iteration)


    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    test_pred_prob += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits
result = np.argmax(test_pred_prob, axis=1)
```
该部分同上面最大的区别就是该表了损失函数和评价函数。分别更换为'multiclass'和'multi_logloss'，当进行多分类任务是必须还要指定类别数：'num_class'。

## 回归任务

```python
params = {'num_leaves': 38,
          'min_data_in_leaf': 50,
          'objective': 'regression',
          'max_depth': -1,
          'learning_rate': 0.02,
          "min_sum_hessian_in_leaf": 6,
          "boosting": "gbdt",
          "feature_fraction": 0.9,
          "bagging_freq": 1,
          "bagging_fraction": 0.7,
          "bagging_seed": 11,
          "lambda_l1": 0.1,
          "verbosity": -1,
          "nthread": 4,
          'metric': 'mae',
          "random_state": 2019,
          # 'device': 'gpu'
          }


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true))) * 100

def smape_func(preds, dtrain):
    label = dtrain.get_label().values
    epsilon = 0.1
    summ = np.maximum(0.5 + epsilon, np.abs(label) + np.abs(preds) + epsilon)
    smape = np.mean(np.abs(label - preds) / summ) * 2
    return 'smape', float(smape), False


folds = KFold(n_splits=5, shuffle=True, random_state=2019)
oof = np.zeros(train_x.shape[0])
predictions = np.zeros(test.shape[0])

train_y = np.log1p(train_y) # Data smoothing
feature_importance_df = pd.DataFrame()
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_x)):
    print("fold {}".format(fold_ + 1))
    trn_data = lgb.Dataset(train_x.iloc[trn_idx], label=train_y.iloc[trn_idx])
    val_data = lgb.Dataset(train_x.iloc[val_idx], label=train_y.iloc[val_idx])


    clf = lgb.train(params,
                    trn_data,
                    num_round,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=200,
                    early_stopping_rounds=200)
    oof[val_idx] = clf.predict(train_x.iloc[val_idx], num_iteration=clf.best_iteration)

    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    predictions += clf.predict(test, num_iteration=clf.best_iteration) / folds.n_splits

print('mse %.6f' % mean_squared_error(train_y, oof))
print('mae %.6f' % mean_absolute_error(train_y, oof))

result = np.expm1(predictions) #reduction
result = predictions
```
在回归任务中对目标函数值添加了一个log平滑，如果待预测的结果值跨度很大，做log平滑很有很好的效果提升。
