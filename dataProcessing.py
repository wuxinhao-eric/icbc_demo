import numpy as np
import pandas as pd

# 读入文件，参数na_values可以设置缺失的数据的内容，在这里不做处理
income = pd.read_excel(r'D:\\icbc_file\\train1.xls')
# 显示所有的列，不会使用省略号代替
pd.set_option('display.max_columns',None)
'''
显示前几行的数据，可以指定显示的行数 income.head()
显示后几行的数据，可以指定显示的行数 income.tail()
整个dataFrame的信息，包含各个列的数据类型和数据量 income.info()
dataFrame的数据类型 income.dtypes
对整个dataFrame的数据描述，可以基本数值统计 income.describe()
检测是否存在缺失值 income.isnull().any()
查看缺失值的统计 income.isnull().sum()
整个dataFrame的数据量 income.count()
查看数据集的行数和列数 income.shape
'''
# 可以观测是否存在异常值
print(income.describe())
# 查看缺失值所占比例，比例大的可以考虑直接删除，使用drop函数'''
print(income.isnull().sum()/income.count())
# 筛选字段
print(income['b_tgt'].value_counts())
print(income[income['b_tgt'] == 0])
# 筛选数据 
print(income.loc[(income['b_tgt'] == 0) & (income['rfm1'] > 20)].head(1))
# 删除两个无用特征列
income.drop(['demog_genm'],axis = 1,inplace = True)
income.drop(['account'],axis = 1,inplace = True)
# 删除重复数据
income.drop_duplicates(inplace=True)
# 缺失值处理:用各自的众数替代
income.fillna(value = {'demog_age':income['demog_age'].mode()[0],
                       'rfm3':income['rfm3'].mean()},
              inplace=True)
# 根据观察发现这个列存在缺失值
income.loc[income['demog_inc'] == 0,'demog_inc'] = np.nan
print(income.isnull().sum())
# income = income.dropna()
# print(income.shape)
income['demog_inc'].fillna(income['demog_inc'].mean(),inplace=True)
print(income.shape)

# 离散变量的重新编码：字符变量转换为整数变量
for feature in income.columns:
    if income[feature].dtype == 'object':
        income[feature] = pd.Categorical(income[feature]).codes

# 异常值检测箱线图法
Q1 = income['demog_age'].quantile(q = 0.25)
Q3 = income['demog_age'].quantile(q = 0.75)
IQR = Q3 - Q1
print('箱线图法异常值上限检测：\n',any(income['demog_age'] > Q3 + 1.5 * IQR))
print('箱线图法异常值下限检测：\n',any(income['demog_age'] < Q1 - 1.5 * IQR))

down_limit = Q1 - 1.5 * IQR
# 删除异常值
# print(income.loc[income['demog_age'] < down_limit].index.tolist())
income.drop(income.loc[income['demog_age'] < down_limit].index.tolist(),inplace=True)
print(income.shape)

income.to_excel('D:\\icbc_file\\train_final.xls', index_label=False)