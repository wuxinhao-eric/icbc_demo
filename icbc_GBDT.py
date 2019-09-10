import pandas as pd
income = pd.read_excel(r'D:\\icbc_file\\train_final.xls')
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(income.loc[:,'cat_input1':'demog_genf'],
                                                 income['b_tgt'],train_size=0.75,test_size=0.25,random_state=1234)

from sklearn.ensemble import GradientBoostingClassifier

gdbt = GradientBoostingClassifier()
gdbt.fit(X_train,y_train)
print(gdbt.score(X_train,y_train))
gdbt_predict = gdbt.predict(X_test)
print(gdbt.score(X_test,y_test))


from sklearn import metrics
# 行是实际，列是预测
cm = metrics.confusion_matrix(y_test,gdbt_predict,labels=[0,1])
print(cm)

Accuracy = metrics.scorer.accuracy_score(y_test,gdbt_predict)
Sensitivity = metrics.scorer.recall_score(y_test,gdbt_predict)
Specificity = metrics.scorer.recall_score(y_test,gdbt_predict,pos_label=0)
print('模型准确率：',Accuracy*100)
print('正例覆盖率：',Sensitivity*100)
print('负例覆盖率：',Specificity*100)
'''
from sklearn import metrics
import matplotlib.pyplot as plt
fpr , tpr , _= metrics.roc_curve(y_test,gdbt.predict_proba(X_test)[:,1])
plt.plot(fpr,tpr,linestyle='solid',color='red')
plt.stackplot(fpr,tpr,color='steelblue')
plt.plot([0,1],[0,1],linestyle='dashed', color='black')
plt.text(0.6,0.4,'AUG=%.3f' % metrics.auc(fpr,tpr),fontdict=dict(size=18))
plt.show()
'''