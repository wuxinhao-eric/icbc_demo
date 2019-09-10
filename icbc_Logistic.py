import pandas as pd
income = pd.read_excel(r'D:\\icbc_file\\train_final.xls')
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(income.loc[:,'cat_input1':'demog_genf'],
                                                 income['b_tgt'],train_size=0.75,test_size=0.25,random_state=1234)

from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,y_train)
print('训练集的准确率：',log.score(X_train,y_train))
log_predict = log.predict(X_test)
print('测试集的准确率：',log.score(X_test,y_test))
print(pd.crosstab(log_predict,y_test))

from sklearn import metrics
# 行是实际，列是预测
cm = metrics.confusion_matrix(y_test,log_predict,labels=[0,1])
print(cm)

Accuracy = metrics.scorer.accuracy_score(y_test,log_predict)
Sensitivity = metrics.scorer.recall_score(y_test,log_predict)
Specificity = metrics.scorer.recall_score(y_test,log_predict,pos_label=0)
print('模型准确率：',Accuracy*100)
print('正例覆盖率：',Sensitivity*100)
print('负例覆盖率：',Specificity*100)