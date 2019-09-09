import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 阶层式分群
income = pd.read_excel(r'D:\\icbc_file\\train_final.xls')
data_x = income.loc[:,'cat_input1':'demog_genf']
data_y = income.loc[:,'b_tgt']

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(data_x,method='ward'))
plt.title('Dendrogram')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='ward')
data = np.array(data_x,dtype=float)
y_hc = hc.fit_predict(data)

# 4:过去三年由于直接营销活动而购买产品的平均销售额
# 2:过去三年平均销售额
# 12:过去一年总共收到的直接营销活动数目
# 17:地区平均房产价值
plt.scatter(data[y_hc == 0,12],data[y_hc == 0,17],s=50,c='red',label='Cluster1')
plt.scatter(data[y_hc == 1,12],data[y_hc == 1,17],s=50,c='blue',label='Cluster2')
# plt.scatter(data[y_hc == 2,4],data[y_hc == 2,2],s=10,c='green',label='Cluster2')
plt.legend()
plt.show()

# 查看实际分类的效果图
data_real = np.array(data_y)
plt.scatter(data[data_real == 0,12],data[data_real == 0,17],s=50,c='red',label='Cluster1')
plt.scatter(data[data_real == 1,12],data[data_real == 1,17],s=50,c='blue',label='Cluster2')
plt.legend()
plt.show()
