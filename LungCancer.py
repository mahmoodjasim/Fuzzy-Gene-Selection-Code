#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data3= pd.read_csv("C:/Users/Mahmood/Desktop/Camparsion/LungCancer/Microarray/DataSetOfLungCancer/GSE19804/SelectedGenes.csv")


# In[5]:


from sklearn import model_selection
import time
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
X = data3.drop('Class', axis=1)#independent columns
X = (X - X.min())/ (X.max() - X.min())
y =data3['Class']
y = LabelEncoder().fit_transform(y)
start = time.time()
i=0
sum1=0
sum2=0
sum3=0
sum4=0
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=0)
    kfold = model_selection.KFold(n_splits=10)
    model1 =  MLPClassifier(hidden_layer_sizes=(300,200,100), activation='relu', 
                                max_iter = 100, solver = 'adam')
    accuracy = cross_val_score(model1, X_train, y_train, cv=kfold,scoring='accuracy')
    sum1=sum1+ np.mean(accuracy)
    recall = cross_val_score(model1, X_train, y_train, cv=kfold, scoring='recall')
    sum2=sum2+ np.mean(recall)
    precision = cross_val_score(model1, X_train, y_train, cv=kfold, scoring='precision')
    sum3=sum3+ np.mean(precision)
    f1 = cross_val_score(model1, X_train, y_train, cv=kfold, scoring='f1')
    sum4=sum4+ np.mean(f1)
print("Overall Accuracy=",sum1/10)
print("Overall Precision=",sum3/10)
print("Overall Recall=",sum2/10)
print("Overall F1-score=",sum4/10)
end = time.time()
print(end - start, "seconds")


# In[7]:


from sklearn import model_selection
import time
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
X = data3.drop('Class', axis=1)#independent columns
y =data3['Class']
y = LabelEncoder().fit_transform(y)
start = time.time()
i=0
sum1=0
sum2=0
sum3=0
sum4=0
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=42)
    kfold = model_selection.KFold(n_splits=10)
    model1 = DecisionTreeClassifier(max_depth=4)
    accuracy = cross_val_score(model1, X_train, y_train, cv=kfold,scoring='accuracy')
    sum1=sum1+ np.mean(accuracy)
    recall = cross_val_score(model1, X_train, y_train, cv=kfold, scoring='recall')
    sum2=sum2+ np.mean(recall)
    precision = cross_val_score(model1, X_train, y_train, cv=kfold, scoring='precision')
    sum3=sum3+ np.mean(precision)
    f1 = cross_val_score(model1, X_train, y_train, cv=kfold, scoring='f1')
    sum4=sum4+ np.mean(f1)
print("Overall Accuracy=",sum1/10)
print("Overall Precision=",sum3/10)
print("Overall Recall=",sum2/10)
print("Overall F1-score=",sum4/10)
end = time.time()
print(end - start, "seconds")


# In[8]:


from sklearn import model_selection
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
X = data3.drop('Class', axis=1)#independent columns
y =data3['Class']
y = LabelEncoder().fit_transform(y)
start = time.time()
i=0
sum1=0
sum2=0
sum3=0
sum4=0
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=42)
    kfold = model_selection.KFold(n_splits=10)
    model1 = KNeighborsClassifier(n_neighbors = 7)
    accuracy = cross_val_score(model1, X_train, y_train, cv=kfold,scoring='accuracy')
    sum1=sum1+ np.mean(accuracy)
    recall = cross_val_score(model1, X_train, y_train, cv=kfold, scoring='recall')
    sum2=sum2+ np.mean(recall)
    precision = cross_val_score(model1, X_train, y_train, cv=kfold, scoring='precision')
    sum3=sum3+ np.mean(precision)
    f1 = cross_val_score(model1, X_train, y_train, cv=kfold, scoring='f1')
    sum4=sum4+ np.mean(f1)
print("Overall Accuracy=",sum1/10)
print("Overall Precision=",sum3/10)
print("Overall Recall=",sum2/10)
print("Overall F1-score=",sum4/10)
end = time.time()
print(end - start, "seconds")


# In[9]:


from sklearn import model_selection
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB 
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
X = data3.drop('Class', axis=1)#independent columns
X = (X - X.min())/ (X.max() - X.min())
y =data3['Class']
y = LabelEncoder().fit_transform(y)
start = time.time()
i=0
sum1=0
sum2=0
sum3=0
sum4=0
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=42)
    kfold = model_selection.KFold(n_splits=10)
    model1 =GaussianNB()
    accuracy = cross_val_score(model1, X_train, y_train, cv=kfold,scoring='accuracy')
    sum1=sum1+ np.mean(accuracy)
    recall = cross_val_score(model1, X_train, y_train, cv=kfold, scoring='recall')
    sum2=sum2+ np.mean(recall)
    precision = cross_val_score(model1, X_train, y_train, cv=kfold, scoring='precision')
    sum3=sum3+ np.mean(precision)
    f1 = cross_val_score(model1, X_train, y_train, cv=kfold, scoring='f1')
    sum4=sum4+ np.mean(f1)
print("Overall Accuracy=",sum1/10)
print("Overall Precision=",sum3/10)
print("Overall Recall=",sum2/10)
print("Overall F1-score=",sum4/10)
end = time.time()
print(end - start, "seconds")


# In[10]:


from sklearn import model_selection
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC 
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
X = data3.drop('Class', axis=1)#independent columns
X = (X - X.min())/ (X.max() - X.min())
y =data3['Class']
y = LabelEncoder().fit_transform(y)
start = time.time()
i=0
sum1=0
sum2=0
sum3=0
sum4=0
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=42)
    kfold = model_selection.KFold(n_splits=10)
    model1 = SVC(kernel = 'linear', C = 1)
    accuracy = cross_val_score(model1, X_train, y_train, cv=kfold,scoring='accuracy')
    sum1=sum1+ np.mean(accuracy)
    recall = cross_val_score(model1, X_train, y_train, cv=kfold, scoring='recall')
    sum2=sum2+ np.mean(recall)
    precision = cross_val_score(model1, X_train, y_train, cv=kfold, scoring='precision')
    sum3=sum3+ np.mean(precision)
    f1 = cross_val_score(model1, X_train, y_train, cv=kfold, scoring='f1')
    sum4=sum4+ np.mean(f1)
print("Overall Accuracy=",sum1/10)
print("Overall Precision=",sum3/10)
print("Overall Recall=",sum2/10)
print("Overall F1-score=",sum4/10)
end = time.time()
print(end - start, "seconds")


# In[ ]:




