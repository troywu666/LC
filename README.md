# 金融信用模型

便签：Xgboost

[TOC]

## 1、背景和目标

背景：对于投资者来说，需要预测借贷者无法偿还贷款的风险，最大程度地避免投资损失，最大程度地实现投资回报。如果借贷者顺利偿还贷款，投资者则获得利息收益。如果借贷者无法偿还贷款，投资者则损失贷款本金

目标：根据贷款记录数据构建机器学习模型，预测借贷者是否有信用问题

## 2、分析方法确定

- 该项目的预测问题为二分类问题，预测用户是否是“坏人”

- 特征未进行脱敏处理，可以利用业务知识对特征进行先处理后再进行常规特征工程

- 在金融领域，容易出现较严重的正负样本不平衡问题，且特征容易出现共线性的情况，因而使用GBRT作为本次问题的模型

## 3、定义taget

```python
df_clean.loan_status.replace('Fully Paid', 1, inplace = True)
df_clean.loan_status.replace('Current', 1, inplace = True)
df_clean.loan_status.replace('Late (16-30 days)', 0, inplace = True)
df_clean.loan_status.replace('Late (31-120 days)', 0, inplace = True)
df_clean.loan_status.replace('Charged Off', 0, inplace = True)
df_clean.loan_status.replace('In Grace Period', np.nan, inplace = True)
df_clean.loan_status.replace('Default', np.nan, inplace = True)

df_clean.dropna(subset = ['loan_status'], inplace = True)
y = df_clean.loan_status
df_clean.drop('loan_status', axis = 1, inplace = True)
```



## 4、数据观察及预处理

### 4.1、清理id类数据以及行与列均为空值的数据

```python
df_clean = df.copy()
df_clean.drop('member_id', axis = 1, inplace = True)
df_clean.drop('id', axis = 1, inplace = True)
df_clean.dropna(axis = 0, how = 'all', inplace = True)
df_clean.dropna(axis = 1, how = 'all', inplace = True)
```

### 4.2、将缺失比例高于50%的特征删除

```python
missing_col = []
missing_col_nums = []
for col in df_clean.columns:
    missing_nums = round(len(df_clean[df_clean[col].isnull()].index) / len(df_clean), 5)
    if missing_nums:
        missing_col.append(col)
        missing_col_nums.append((col, missing_nums))
for i in sorted(missing_col_nums, key = lambda x: x[1], reverse = True):
    print(i)
    if i[1] >0.5:
        missing_col.remove(i[0])
        missing_col_nums.remove(i)
        df_clean.drop(i[0], axis = 1, inplace = True)
```

### 4.3、将object型数据取值个数超过50的特征删除

```python
object_list = []
object_col = []
for col in df_clean.select_dtypes(include = 'object').columns:
    object_list.append((col, len(df_clean[col].unique())))
    object_col.append(col)
for i in sorted(object_list, key = lambda x: x[1], reverse = True):
    print(i)
    if i[1] >= 50:
        object_list.remove(i)
        object_col.remove(i[0])
        df_clean.drop(i[0], axis = 1, inplace = True)
```

### 4.4、部分object数据处理为数值型

```
df_clean.emp_length.fillna(0, inplace = True)
df_clean.emp_length.replace(to_replace = '[^0-9]+', value = '', inplace = True, regex = True)
df_clean.emp_length = df_clean.emp_length.astype(int)

#将百分比数据处理成int型
df_clean.revol_util = df_clean.revol_util.str.replace('%', "").astype(float)
df_clean.int_rate = df_clean.int_rate.str.replace('%','').astype(float)
```

### 4.5、删除贷后特征

```python
df.drop(['out_prncp','out_prncp_inv','total_pymnt',
         'total_pymnt_inv','total_rec_prncp', 'grade', 'sub_grade'] ,1, inplace=True)
df.drop(['total_rec_int','total_rec_late_fee',
         'recoveries','collection_recovery_fee',
         'collection_recovery_fee' ],1, inplace=True)
df.drop(['last_pymnt_d','last_pymnt_amnt',
         'next_pymnt_d','last_credit_pull_d'],1, inplace=True)
df.drop(['policy_code'],1, inplace=True)
```

### 4.6、删除方差小于3的数值型特征

```python
df_clean.select_dtypes(include = ['float']).describe().T.sort_values(by = 'std')

float_col = list(df_clean.select_dtypes(include = ['float']).columns)
for col in float_col:
    if df_clean[col].std() <= 3:
        df_clean.drop(col, axis = 1, inplace = True)
```

### 4.7、相关性较强特征剔除

```python
cor = df_clean.corr()
cor.loc[ : , : ] = np.tril(df_clean.corr().round(5), k = -1)
cor = cor.stack()
cor_set = set()
for index in cor[(cor > 0.6)].index:
    cor_set.add(index[0])
for col in cor_set:
    df_clean.drop(col, axis = 1, inplace = True)
```

### 4.8、缺失数据的相关性分析

![png](E:\Data Analyse\Course\Practice\风控\README\missing_heatmap.png)

- mo_sin_old_il_acct与mths_since_rcnt_il的缺失情况完全正相关，后续需要为其构建一个特征

### 4.9、