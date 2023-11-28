来源：[【lightgbm/xgboost/nn代码整理二】xgboost做二分类，多分类以及回归任务](https://zhuanlan.zhihu.com/p/78777407)
## 简单的数据处理

```Python
data = pd.concat([train_data, test_data])  # 沿着行的方向连接训练数据和测试数据集
cate_feature = ['gender', 'cell_province', 'id_province', 'id_city', 'rate', 'term']  # 定义数据集中的分类特征的列表
for item in cate_feature:
    data[item] = LabelEncoder().fit_transform(data[item])  # 对每个分类特征应用标签编码，它为每个类别分配一个唯一的整数。
    item_dummies = pd.get_dummies(data[item])  # 使用独热编码为分类特征创建虚拟变量。每个唯一值在分类特征中都有自己的二进制列。
    item_dummies.columns = [item + str(i + 1) for i in range(item_dummies.shape[1])]  # 将虚拟变量的列重命名为包含原始特征名和数字后缀的格式。
    data = pd.concat([data, item_dummies], axis=1)  # 沿着列的方向将虚拟变量连接到原始数据集。
data.drop(cate_feature,axis=1,inplace=True)  # 从数据集中删除原始的分类特征，因为它们已经被它们的独热编码替代。
```
在工程中，如果类别过多，我一般会放弃进行onehot，主要是由于进行onehot会导致特征过于稀疏，运算速度变慢，严重影响模型的迭代速度，并且最终对结果提升很有限,我通常只会进行labelEncoder, 也可以对特征进行embeding处理。

## 模型
### 参数

以二分类为例
```python
params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'gamma': 0.1,
    'max_depth': 8,
    'alpha': 0,
    'lambda': 0,
    'subsample': 0.7,
    'colsample_bytree': 0.5,
    'min_child_weight': 3,
    'silent': 0,
    'eta': 0.03,
    'nthread': -1,
    'seed': 2019,
}
```

* objective(string or callable)：在回归问题一般使用reg:squarederror ，即MSE均方误差。二分类问题一般使用binary:logistic, 多分类问题一般使用multi:softmax,当是多分类任务时需要指定类别数量，eg:'num_class':33。
* eval_metric (str, list of str, or callable, optional, default=None):指定模型评估指标。常见的值包括：'rmse'：均方根误差（用于回归问题）。'mae'：平均绝对误差（用于回归问题）。'logloss'：对数损失（用于二分类问题）。'auc'也常用于二分类问题。'mlogloss'常用于多分类问题。'error'：错误率（用于分类问题）。
* gamma (float, optional, default=0):用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子
* max_depth (int, optional, default=6):每棵树的最大深度。较大的深度可能会导致过拟合。
* alpha (float, optional, default=0):L1 正则化权重。用于控制模型复杂度。
* lambda (float, optional, default=1):L2 正则化权重。用于控制模型复杂度。
* subsample (float, optional, default=1):训练每棵树时使用的样本比例。通俗理解就是选多少样本做为训练集，选择小于1的比例可以减少方差，即防止过拟合。
* colsample_bytree (float, optional, default=1):每棵树的特征（列）采样比例。选择多少列作为训练集，具体的理解就是选择多少特征
* min_child_weight (float, optional, default=1):叶子节点的最小样本权重和。用于控制过拟合，较大的值使算法更保守。
* eta:学习率，控制每次提升的步长。较小的学习率通常需要较大的 n_estimators
* silent: 是否打印训练过程中的信息，0表示打印，1反之
* nthread：运行的线程数，-1所有线程，该值需要根据具体情况调整，线程对最终结果有一点影响，曾今测试，线程越多，结果会变差一丢丢
* seed：这个随机指定一个常数，防止每次结果不一致

  
* base_score (float, optional, default=0.5):在计算初始的预测分数时使用的基准分数。
* booster (string, optional, default='gbtree'):同框架参数，指定使用的弱提升器类型。可以是默认的gbtree, 也就是CART决策树，还可以是线性弱学习器gblinear以及DART。一般来说，我们使用gbtree就可以了，不需要调参。
* n_estimators：是非常重要的要调的参数，它关系到我们XGBoost模型的复杂度，因为它代表了我们决策树弱学习器的个数。这个参数对应sklearn GBDT的n_estimators。n_estimators太小，容易欠拟合，n_estimators太大，模型会过于复杂，一般需要调参选择一个适中的数值。
* scale_pos_weight (float, optional, default=1):在类别不平衡问题中，控制正负权重的平衡。
* n_jobs (int, optional, default=-1):并行线程数，用于拟合和预测。默认值 -1 表示使用所有可用的 CPU 核心。

### 5折交叉

```python
folds = KFold(n_splits=5, shuffle=True, random_state=2019)
```
采用五折交叉统计实际就是训练多个模型和平均值融合，如果时间允许的情况下10折交叉会好于5折。5折交叉还可以采用StratifiedKFold做切分。

### 数据加载

XGBoost可以加载多种数据格式的训练数据：libsvm 格式的文本数据、Numpy 的二维数组、XGBoost 的二进制的缓存文件。加载的数据存储在对象 DMatrix 中，而llightgbm是存储在Dataset中
```python
trn_data = xgb.DMatrix(train_x.iloc[trn_idx], label=train_y[trn_idx])
val_data = xgb.DMatrix(train_x.iloc[val_idx], label=train_y[val_idx])
```

### 训练和预测

```python
##训练部分
watchlist = [(trn_data, 'train'), (val_data, 'valid')]
clf = xgb.train(params, trn_data, num_round, watchlist, verbose_eval=200, early_stopping_rounds=200)

##预测部分
test_pred_prob += clf.predict(xgb.DMatrix(test), ntree_limit=clf.best_ntree_limit) / folds.n_splits
```
* params：参数字典
* trn_data ：训练的数据
* num_round：迭代次数
* watchlist：这是一个列表，用于对训练过程中进行评估列表中的元素。形式是evals =[(dtrain,’train’),(dval,’val’)]或者是evals =[(dtrain,’train’)],对于第一种情况，它使得我们可以在训练过程中观察验证集的效果。
* verbose_eval： 如果为True ,则对evals中元素的评估结果会输出在结果中；如果输入数字，假设为5，则每隔5个迭代输出一次。
* ntree_limit：验证集中最好的结果做预测

## 模型重要性

模型重要性是根据树模型中该特征的分裂次数做统计的，可以基于此重要性来判断特种的重要程度，深入的挖掘特征，具体代码如下：
```python
##保存特征重要性
fold_importance_df = pd.DataFrame()
fold_importance_df["Feature"] = clf.get_fscore().keys()
fold_importance_df["importance"] = clf.get_fscore().values()
fold_importance_df["fold"] = fold_ + 1
feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

##特征重要性显示
## plot feature importance
cols = (feature_importance_df[["Feature", "importance"]] 
        .groupby("Feature").mean().
        sort_values(by="importance", ascending=False).index)
best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]
                                .sort_values(by='importance',ascending=False)
plt.figure(figsize=(8, 15))
sns.barplot(y="Feature", x="importance",
            data=best_features.sort_values(by="importance", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('../../result/xgb_importances.png')
```
在lightgbm中对应的事clf.feature_importance()函数，而在xgboost中对应的是clf.get_fscore()函数。如果特征过多无法完成显示，可以只取topN显示，如只显示top5
```python
cols = (feature_importance_df[["Feature", "importance"]].groupby("Feature").mean()
        .sort_values(by="importance", ascending=False)[:5].index)
```

## 总结

xgboost和lightgbm对比，它的速度会慢很多，使用也没有lighgbm方便，但是可以将xgboost训练的结果和lightgbm做融合，提升最终的结果。

代码地址：[data_mining_models](https://github.com/QLMX/data_mining_models)
