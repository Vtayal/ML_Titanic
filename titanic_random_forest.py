import sklearn.linear_model
import sklearn.model_selection
import sklearn.tree
import sklearn.ensemble
import pandas as pd
import numpy as np

data=pd.read_csv("D://ML bvp//Data//train.csv")
#print(data.groupby(["Species"]).mean(),'\n')
#print(data["Species"].unique())
data=data.drop(["Cabin","PassengerId","Name","Fare","Ticket"],axis=1)
data=data.join(pd.get_dummies(data["Sex"]))
data=data.drop(["Sex"],axis=1)
data=data.join(pd.get_dummies(data["Embarked"]))
data=data.drop(["Embarked"],axis=1)
data=data.dropna()
y=data["Survived"]
data=data.drop(["Survived"],axis=1)
x=data

#print(y)
#print(x.head())
#print(x.head(),'\n')
#print(y.head(),'\n')

x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.2)

clf=sklearn.ensemble.RandomForestClassifier(random_state=0)
clf.fit(x_train,y_train)
#clf_gini.fit(x_test,y_test)

print('train=',clf.score(x_train,y_train),'\n')
print('test=',clf.score(x_test,y_test),'\n')
#print(clf.feature_importances_,'\n')
#print(x.head())

data1=pd.read_csv("D://ML bvp//Data//test.csv")
data2=pd.read_csv("D://ML bvp//Data//test.csv")
#print(data.groupby(["Species"]).mean(),'\n')
#print(data["Species"].unique())
data1=data1.drop(["Cabin","PassengerId","Name","Fare","Ticket"],axis=1)
data1=data1.join(pd.get_dummies(data1["Sex"]))
data1=data1.drop(["Sex"],axis=1)
data1=data1.join(pd.get_dummies(data1["Embarked"]))
data1=data1.drop(["Embarked"],axis=1)
data1=data1.dropna()
x1=data1
print(x1.head())
clf.fit(x,y)

data2=data2.drop(["Cabin","Name","Fare","Ticket"],axis=1)
data2=data2.join(pd.get_dummies(data2["Sex"]))
data2=data2.drop(["Sex"],axis=1)
data2=data2.join(pd.get_dummies(data2["Embarked"]))
data2=data2.drop(["Embarked"],axis=1)
data2=data2.dropna()

ans=clf.predict(x1)
print(len(ans),'\n\n\n\n\n',data2["PassengerId"])

#Write to CSV
submission =pd.DataFrame({"Id":data2["PassengerId"],"Survived":ans})
submission.to_csv('Titanic_Predicted.csv',index=False)
#sklearn.tree.export_graphviz(clf_gini,out_file='titanic_.dot')
#print(sklearn.tree.export_graphviz(clf_gini,out_file='titanic_.dot'))