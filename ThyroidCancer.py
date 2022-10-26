#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
data3 = pd.read_csv("C:/Users/Mahmood/Desktop/Camparsion/thyroid/Microarray/FuzzyGenes.csv")


# In[2]:


data3['Class'].value_counts()


# In[2]:


from sklearn import model_selection
from sklearn.model_selection import cross_validate
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
X = data3.drop('Class', axis=1)#independent columns
y =data3['Class']
y = LabelEncoder().fit_transform(y)
scoring = ['precision_macro', 'recall_macro','f1_macro']
start = time.time()
i=0
j=0
a=[]
b=[]
c=[]
d=[]
sum1=0
sum2=0
sum3=0
sum4=0
overalAcc=0
overalPre=0
overalRe=0
oVeralF1=0
for j in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=0)
    kfold = model_selection.KFold(n_splits=10)
    model1 = DecisionTreeClassifier(max_depth=4)
    accuracy = cross_validate(model1, X_train, y_train,cv=kfold, scoring="accuracy")
    scores = cross_validate(model1, X_train, y_train,cv=kfold, scoring=scoring)
    end = time.time()
    a=scores['test_precision_macro']
    b=scores['test_recall_macro']
    c=scores['test_f1_macro']
    d=accuracy['test_score']
    for i in range (10):
        sum1=sum1+a[i]
        sum2=sum2+b[i]
        sum3=sum3+c[i]
        sum4=sum4+d[i]
    overalAcc=overalAcc+sum4/10
    overalPre=overalPre+sum1/10
    overalRe=overalRe+sum2/10
    oVeralF1=oVeralF1+sum3/10
    sum1=0
    sum2=0
    sum3=0
    sum4=0
print("OverAll Accuracy=",overalAcc/10)
print("OverAll Precision=",overalPre/10)
print("OverAll Recall=",overalRe/10)
print("OverAll F1-score=",oVeralF1/10)
print("Time Taken is",end - start, "seconds")


# In[15]:


from sklearn import model_selection
from sklearn.model_selection import cross_validate
import time
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
X = data3.drop('Class', axis=1)#independent columns
X = (X - X.min())/ (X.max() - X.min())
y =data3['Class']
y = LabelEncoder().fit_transform(y)
scoring = ['precision_macro', 'recall_macro','f1_macro']
start = time.time()
i=0
j=0
a=[]
b=[]
c=[]
d=[]
sum1=0
sum2=0
sum3=0
sum4=0
overalAcc=0
overalPre=0
overalRe=0
oVeralF1=0
for j in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=0)
    kfold = model_selection.KFold(n_splits=10)
    model1 =  MLPClassifier(hidden_layer_sizes=(300,200,100), activation='relu', 
                                solver = 'adam', alpha = 0.0001, learning_rate = 'constant',learning_rate_init = 0.001)
    accuracy = cross_validate(model1, X_train, y_train,cv=kfold, scoring="accuracy")
    scores = cross_validate(model1, X_train, y_train,cv=kfold, scoring=scoring)
    end = time.time()
    a=scores['test_precision_macro']
    b=scores['test_recall_macro']
    c=scores['test_f1_macro']
    d=accuracy['test_score']
    for i in range (10):
        sum1=sum1+a[i]
        sum2=sum2+b[i]
        sum3=sum3+c[i]
        sum4=sum4+d[i]
    overalAcc=overalAcc+sum4/10
    overalPre=overalPre+sum1/10
    overalRe=overalRe+sum2/10
    oVeralF1=oVeralF1+sum3/10
    sum1=0
    sum2=0
    sum3=0
    sum4=0
print("Accuracy=",overalAcc/10)
print("Precision=",overalPre/10)
print("Recall=",overalRe/10)
print("F1-score=",oVeralF1/10)
print("Time Taken is",end - start, "seconds")


# In[4]:


from sklearn import model_selection
from sklearn.model_selection import cross_validate
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
X = data3.drop('Class', axis=1)#independent columns
y =data3['Class']
y = LabelEncoder().fit_transform(y)
scoring = ['precision_macro', 'recall_macro','f1_macro']
start = time.time()
i=0
j=0
a=[]
b=[]
c=[]
d=[]
sum1=0
sum2=0
sum3=0
sum4=0
overalAcc=0
overalPre=0
overalRe=0
oVeralF1=0
for j in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=42)
    kfold = model_selection.KFold(n_splits=10)
    model1 =  KNeighborsClassifier(n_neighbors = 7)
    accuracy = cross_validate(model1, X_train, y_train,cv=kfold, scoring="accuracy")
    scores = cross_validate(model1, X_train, y_train,cv=kfold, scoring=scoring)
    end = time.time()
    a=scores['test_precision_macro']
    b=scores['test_recall_macro']
    c=scores['test_f1_macro']
    d=accuracy['test_score']
    for i in range (10):
        sum1=sum1+a[i]
        sum2=sum2+b[i]
        sum3=sum3+c[i]
        sum4=sum4+d[i]
    overalAcc=overalAcc+sum4/10
    overalPre=overalPre+sum1/10
    overalRe=overalRe+sum2/10
    oVeralF1=oVeralF1+sum3/10
    sum1=0
    sum2=0
    sum3=0
    sum4=0
print("Accuracy=",overalAcc/10)
print("Precision=",overalPre/10)
print("Recall=",overalRe/10)
print("F1-score=",oVeralF1/10)
print("Time Taken is",end - start, "seconds")


# In[5]:


from sklearn import model_selection
from sklearn.model_selection import cross_validate
import time
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
X = data3.drop('Class', axis=1)#independent columns
y =data3['Class']
y = LabelEncoder().fit_transform(y)
scoring = ['precision_macro', 'recall_macro','f1_macro']
start = time.time()
i=0
j=0
a=[]
b=[]
c=[]
d=[]
sum1=0
sum2=0
sum3=0
sum4=0
overalAcc=0
overalPre=0
overalRe=0
oVeralF1=0
for j in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=42)
    kfold = model_selection.KFold(n_splits=10)
    model1 =  GaussianNB()
    accuracy = cross_validate(model1, X_train, y_train,cv=kfold, scoring="accuracy")
    scores = cross_validate(model1, X_train, y_train,cv=kfold, scoring=scoring)
    end = time.time()
    a=scores['test_precision_macro']
    b=scores['test_recall_macro']
    c=scores['test_f1_macro']
    d=accuracy['test_score']
    for i in range (10):
        sum1=sum1+a[i]
        sum2=sum2+b[i]
        sum3=sum3+c[i]
        sum4=sum4+d[i]
    overalAcc=overalAcc+sum4/10
    overalPre=overalPre+sum1/10
    overalRe=overalRe+sum2/10
    oVeralF1=oVeralF1+sum3/10
    sum1=0
    sum2=0
    sum3=0
    sum4=0
print("Accuracy=",overalAcc/10)
print("Precision=",overalPre/10)
print("Recall=",overalRe/10)
print("F1-score=",oVeralF1/10)
print("Time Taken is",end - start, "seconds")


# In[6]:


from sklearn import model_selection
from sklearn.model_selection import cross_validate
import time
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
X = data3.drop('Class', axis=1)#independent columns
y =data3['Class']
y = LabelEncoder().fit_transform(y)
scoring = ['precision_macro', 'recall_macro','f1_macro']
start = time.time()
i=0
j=0
a=[]
b=[]
c=[]
d=[]
sum1=0
sum2=0
sum3=0
sum4=0
overalAcc=0
overalPre=0
overalRe=0
oVeralF1=0
for j in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=42)
    kfold = model_selection.KFold(n_splits=10)
    model1 =  SVC(kernel = 'linear', C = 1)
    accuracy = cross_validate(model1, X_train, y_train,cv=kfold, scoring="accuracy")
    scores = cross_validate(model1, X_train, y_train,cv=kfold, scoring=scoring)
    end = time.time()
    a=scores['test_precision_macro']
    b=scores['test_recall_macro']
    c=scores['test_f1_macro']
    d=accuracy['test_score']
    for i in range (10):
        sum1=sum1+a[i]
        sum2=sum2+b[i]
        sum3=sum3+c[i]
        sum4=sum4+d[i]
    overalAcc=overalAcc+sum4/10
    overalPre=overalPre+sum1/10
    overalRe=overalRe+sum2/10
    oVeralF1=oVeralF1+sum3/10
    sum1=0
    sum2=0
    sum3=0
    sum4=0
print("Accuracy=",overalAcc/10)
print("Precision=",overalPre/10)
print("Recall=",overalRe/10)
print("F1-score=",oVeralF1/10)
print("Time Taken is",end - start, "seconds")


# In[ ]:




