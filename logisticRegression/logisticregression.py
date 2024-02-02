import numpy as np
import matplotlib as mp 
import pandas as pd 

veriler = pd.read_csv("veriler.csv")

x=veriler.iloc[:,2:3]
y=veriler.iloc[:,4:]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.33)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)

from sklearn.linear_model import LogisticRegression

logr=LogisticRegression(random_state=0)
logr.fit(x_train,y_train)

y_pred=logr.predict(X_test)
print(y_pred)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print(cm)





