import pandas as pd
income = pd.read_excel(r'D:\\icbc_file\\train_final.xls')
from sklearn.model_selection import train_test_split
# 数据拆分
X_train,X_test,y_train,y_test = train_test_split(income.loc[:,'cat_input1':'demog_genf'],
                                                 income['b_tgt'],train_size=0.75,test_size=0.25,random_state=1234)

print('训练数据集共有%d条观测' %X_train.shape[0])
print('测试数据集共有%d条观测' %X_test.shape[0])

from sklearn.neighbors import KNeighborsClassifier
# 默认的k值为5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
knn_predict = knn.predict(X_test)
# 查看分类结果
print(pd.Series(knn_predict,y_test).value_counts())
# 混淆矩阵
print(pd.crosstab(knn_predict,y_test))
print('训练集的准确率：',knn.score(X_train,y_train))
print('测试集的准确率：',knn.score(X_test,y_test))
'''
# k近邻模型的网络搜索法
# 导入网络搜索法的函数
from sklearn.model_selection import GridSearchCV
# 选择不同的参数
param_grid=[{
    "weights":["uniform"],
    "n_neighbors":[i for i in range(1,11)]
},
{
    "weights":["distance"],
    "n_neighbors":[i for i in range(1,11)],
    "p":[i for i in range(1,6)]
}]
knn_clf = KNeighborsClassifier()
grid_kn = GridSearchCV(knn_clf,param_grid,verbose=2,scoring='accuracy')
grid_kn.fit(X_train, y_train)

# 网格搜索到的最佳分类器参数
print(grid_kn.best_estimator_)
# 最佳指定的参数
print(grid_kn.best_params_)
print(grid_kn.grid_scores_)

# 最优的准确率和最优的k值
best_rate = 0
best_k = 0
# for循环是最基础的笨办法
for i in range(1,11):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    rate = knn.score(X_test, y_test)

    if rate > best_rate:
        best_rate = rate
        best_k = i

print('最优的k:',best_k)
print('最高的准确率:',best_rate)
'''

# 评测算法
from sklearn import metrics
import matplotlib.pyplot as plt
# 计算ROC曲线的x轴和y轴数据
fpr , tpr , _= metrics.roc_curve(y_test,knn.predict_proba(X_test)[:,1])
# 绘制曲线
plt.plot(fpr,tpr,linestyle='solid',color='red')
# 添加阴影
plt.stackplot(fpr,tpr,color='steelblue')
# 绘制参考线
plt.plot([0,1],[0,1],linestyle='dashed', color='black')
# 添加文本
plt.text(0.6,0.4,'AUG=%.3f' % metrics.auc(fpr,tpr),fontdict=dict(size=18))
plt.show()