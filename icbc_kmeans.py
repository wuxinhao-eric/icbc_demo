'''
Kmeans进行数据集的聚类
主要选用训练集中的RFM-5到RFM-12的几个维度来作为依据
'''
import pandas as pd
import matplotlib.pyplot as plt
income = pd.read_excel(r'D:\\icbc_file\\train1.xls')
# 查看数据集的前五行 : print(income.head())
import seaborn as sns
# 横轴：生命周期长度 纵轴：整个生命周期购买产品数量
sns.lmplot(x = 'rfm12',y = 'rfm6',data=income,
           fit_reg=False,scatter_kws={'alpha':0.1,'color':'red'})
plt.show()

# 数据标准化处理
from sklearn import preprocessing
X = preprocessing.minmax_scale(income[['rfm1','rfm2','rfm5','rfm6','rfm12']])
# 将数据转换为数据框
X = pd.DataFrame(X,columns=['rfm1','rfm2','rfm5','rfm6','rfm12'])
# 使用拐点法选择最佳的K值
from sklearn.cluster import KMeans
distortions = []
for i in range(1,11):
    km = KMeans(n_clusters=i,init="k-means++",n_init=10,max_iter=300,tol=1e-4,random_state=0)
    km.fit(X)
    #获取K-means算法的SSE
    distortions.append(km.inertia_)
#绘制曲线
plt.plot(range(1,11),distortions,marker="o")
plt.xlabel('簇数量')
plt.ylabel('簇内误方差(SSE)')
plt.show()

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
income['cluster'] = kmeans.labels_

print(income.cluster.unique())
centres = []
for i in income.cluster.unique():
    centres.append(income.loc[income.cluster==i,['rfm1','rfm2','rfm5','rfm6','rfm12']].mean())

import numpy as np
centres = np.array(centres)

sns.lmplot(x = 'rfm12',y = 'rfm6',hue='cluster',data=income,markers='^',
           fit_reg=False,scatter_kws={'alpha':0.8},legend=False)
plt.scatter(centres[:,0],centres[:,2],c='k',marker='*',s=180)
plt.xlabel('生命周期长度')
plt.ylabel('整个生命周期购买产品数量')
plt.show()