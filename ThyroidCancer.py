#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
data3 = pd.read_csv("C:/Users/Mahmood/Desktop/Camparsion/thyroid/Microarray/FuzzyGenes.csv")


# In[3]:


data3['Class'].value_counts()


# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import model_selection
import time
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
X = data3.drop('Class', axis=1)#independent columns
y =data3['Class']
scoring = ['precision_macro', 'recall_macro','f1_macro']
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y)
label_encoder_name_mapping = dict(zip(label_encoder.classes_,
                                         label_encoder.transform(label_encoder.classes_)))
# K-Fold Cross-Validation
from sklearn.model_selection import cross_validate
def cross_validation(model, _X, _y, _cv=10):
    sum1=0
    sum2=0
    sum3=0
    a=[]
    b=[]
    c=[]
    _scoring = ['precision_macro', 'recall_macro', 'f1_macro']
    results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)
    
    a=results['test_precision_macro']
    b=results['test_recall_macro']
    c=results['test_f1_macro']
    for i in range (10):
        sum1=sum1+a[i]
        sum2=sum2+b[i]
        sum3=sum3+c[i]
    print("Precision=",sum1/10)
    print("Recall=",sum2/10)
    print("F1-score=",sum3/10)
    
              
def cross_validation1(model, _X, _y, _cv=10):
    d=[]
    sum4=0
    results1 = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring='accuracy',
                               return_train_score=True)
      
    
    d=results1['test_score']
    for j in range (10):
        sum4=sum4+d[j]
    print("Accuracy=",sum4/10)
              
    # Grouped Bar Chart for both training and validation data
def plot_result(x_label, y_label, plot_title, train_data, val_data):
       
        # Set size of plot
        plt.figure(figsize=(12,6))
        labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.40000, 1)
        plt.bar(X_axis-0.2, train_data, 0.4, color='blue', label='Training')
        plt.bar(X_axis+0.2, val_data, 0.4, color='red', label='Validation')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()
SVM = SVC(kernel = 'rbf')
cross_validation1(SVM,X,encoded_y,10)
cross_validation(SVM, X, encoded_y, 10)


# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import model_selection
import time
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
X = data3.drop('Class', axis=1)#independent columns
X = (X - X.min())/ (X.max() - X.min())
y =data3['Class']
scoring = ['precision_macro', 'recall_macro','f1_macro']
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y)
label_encoder_name_mapping = dict(zip(label_encoder.classes_,
                                         label_encoder.transform(label_encoder.classes_)))
# K-Fold Cross-Validation
from sklearn.model_selection import cross_validate
def cross_validation(model, _X, _y, _cv=10):
    sum1=0
    sum2=0
    sum3=0
    a=[]
    b=[]
    c=[]
    _scoring = ['precision_macro', 'recall_macro', 'f1_macro']
    results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)
    
    a=results['test_precision_macro']
    b=results['test_recall_macro']
    c=results['test_f1_macro']
    for i in range (10):
        sum1=sum1+a[i]
        sum2=sum2+b[i]
        sum3=sum3+c[i]
    print("Precision=",sum1/10)
    print("Recall=",sum2/10)
    print("F1-score=",sum3/10)
    
              
def cross_validation1(model, _X, _y, _cv=10):
    d=[]
    sum4=0
    results1 = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring='accuracy',
                               return_train_score=True)
      
    
    d=results1['test_score']
    for j in range (10):
        sum4=sum4+d[j]
    print("Accuracy=",sum4/10)
              
    # Grouped Bar Chart for both training and validation data
def plot_result(x_label, y_label, plot_title, train_data, val_data):
       
        # Set size of plot
        plt.figure(figsize=(12,6))
        labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.40000, 1)
        plt.bar(X_axis-0.2, train_data, 0.4, color='blue', label='Training')
        plt.bar(X_axis+0.2, val_data, 0.4, color='red', label='Validation')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()
MLP = MLPClassifier(hidden_layer_sizes=(300,200,100), activation='relu', 
                                max_iter = 200, solver = 'adam',random_state=0)
cross_validation1(MLP,X,encoded_y,10)
cross_validation(MLP, X, encoded_y, 10)


# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import model_selection
import time
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
X = data3.drop('Class', axis=1)#independent columns
X = (X - X.min())/ (X.max() - X.min())
y =data3['Class']
scoring = ['precision_macro', 'recall_macro','f1_macro']
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y)
label_encoder_name_mapping = dict(zip(label_encoder.classes_,
                                         label_encoder.transform(label_encoder.classes_)))
# K-Fold Cross-Validation
from sklearn.model_selection import cross_validate
def cross_validation(model, _X, _y, _cv=10):
    sum1=0
    sum2=0
    sum3=0
    a=[]
    b=[]
    c=[]
    _scoring = ['precision_macro', 'recall_macro', 'f1_macro']
    results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)
    
    a=results['test_precision_macro']
    b=results['test_recall_macro']
    c=results['test_f1_macro']
    for i in range (10):
        sum1=sum1+a[i]
        sum2=sum2+b[i]
        sum3=sum3+c[i]
    print("Precision=",sum1/10)
    print("Recall=",sum2/10)
    print("F1-score=",sum3/10)
    
              
def cross_validation1(model, _X, _y, _cv=10):
    d=[]
    sum4=0
    results1 = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring='accuracy',
                               return_train_score=True)
      
    
    d=results1['test_score']
    for j in range (10):
        sum4=sum4+d[j]
    print("Accuracy=",sum4/10)
              
    # Grouped Bar Chart for both training and validation data
def plot_result(x_label, y_label, plot_title, train_data, val_data):
       
        # Set size of plot
        plt.figure(figsize=(12,6))
        labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.40000, 1)
        plt.bar(X_axis-0.2, train_data, 0.4, color='blue', label='Training')
        plt.bar(X_axis+0.2, val_data, 0.4, color='red', label='Validation')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()
KNN =  KNeighborsClassifier(n_neighbors = 7)
cross_validation1(KNN,X,encoded_y,10)
cross_validation(KNN, X, encoded_y, 10)


# In[38]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import model_selection
import time
from sklearn.naive_bayes import GaussianNB 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
X = data3.drop('Class', axis=1)#independent columns
X = (X - X.min())/ (X.max() - X.min())
y =data3['Class']
scoring = ['precision_macro', 'recall_macro','f1_macro']
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y)
label_encoder_name_mapping = dict(zip(label_encoder.classes_,
                                         label_encoder.transform(label_encoder.classes_)))
# K-Fold Cross-Validation
from sklearn.model_selection import cross_validate
def cross_validation(model, _X, _y, _cv=10):
    sum1=0
    sum2=0
    sum3=0
    a=[]
    b=[]
    c=[]
    _scoring = ['precision_macro', 'recall_macro', 'f1_macro']
    results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)
    
    a=results['test_precision_macro']
    b=results['test_recall_macro']
    c=results['test_f1_macro']
    for i in range (10):
        sum1=sum1+a[i]
        sum2=sum2+b[i]
        sum3=sum3+c[i]
    print("Precision=",sum1/10)
    print("Recall=",sum2/10)
    print("F1-score=",sum3/10)
    
              
def cross_validation1(model, _X, _y, _cv=10):
    d=[]
    sum4=0
    results1 = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring='accuracy',
                               return_train_score=True)
      
    
    d=results1['test_score']
    for j in range (10):
        sum4=sum4+d[j]
    print("Accuracy=",sum4/10)
              
    # Grouped Bar Chart for both training and validation data
def plot_result(x_label, y_label, plot_title, train_data, val_data):
       
        # Set size of plot
        plt.figure(figsize=(12,6))
        labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.40000, 1)
        plt.bar(X_axis-0.2, train_data, 0.4, color='blue', label='Training')
        plt.bar(X_axis+0.2, val_data, 0.4, color='red', label='Validation')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()
GNB =GaussianNB()
cross_validation1(GNB,X,encoded_y,10)
cross_validation(GNB, X, encoded_y, 10)


# In[36]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import model_selection
import time
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
X = data3.drop('Class', axis=1)#independent columns
X = (X - X.min())/ (X.max() - X.min())
y =data3['Class']
scoring = ['precision_macro', 'recall_macro','f1_macro']
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y)
label_encoder_name_mapping = dict(zip(label_encoder.classes_,
                                         label_encoder.transform(label_encoder.classes_)))
# K-Fold Cross-Validation
from sklearn.model_selection import cross_validate
def cross_validation(model, _X, _y, _cv=10):
    sum1=0
    sum2=0
    sum3=0
    a=[]
    b=[]
    c=[]
    _scoring = ['precision_macro', 'recall_macro', 'f1_macro']
    results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)
    
    a=results['test_precision_macro']
    b=results['test_recall_macro']
    c=results['test_f1_macro']
    for i in range (10):
        sum1=sum1+a[i]
        sum2=sum2+b[i]
        sum3=sum3+c[i]
    print("Precision=",sum1/10)
    print("Recall=",sum2/10)
    print("F1-score=",sum3/10)
    
              
def cross_validation1(model, _X, _y, _cv=10):
    d=[]
    sum4=0
    results1 = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring='accuracy',
                               return_train_score=True)
      
    
    d=results1['test_score']
    for j in range (10):
        sum4=sum4+d[j]
    print("Accuracy=",sum4/10)
              
    # Grouped Bar Chart for both training and validation data
def plot_result(x_label, y_label, plot_title, train_data, val_data):
       
        # Set size of plot
        plt.figure(figsize=(12,6))
        labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.40000, 1)
        plt.bar(X_axis-0.2, train_data, 0.4, color='blue', label='Training')
        plt.bar(X_axis+0.2, val_data, 0.4, color='red', label='Validation')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()
DT = DecisionTreeClassifier(max_depth=4,random_state=0)
cross_validation1(DT,X,encoded_y,10)
cross_validation(DT, X, encoded_y, 10)


# In[ ]:




