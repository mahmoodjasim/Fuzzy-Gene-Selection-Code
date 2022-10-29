#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
data3= pd.read_csv("C:/Users/Mahmood/Desktop/Camparsion/LiverCancer/RNA-seq/LiverCancer_GSE77314/FuzzyGenes.csv")


# In[3]:


data3['Class'].value_counts()


# In[3]:


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
y =data3['Class']
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y)
label_encoder_name_mapping = dict(zip(label_encoder.classes_,
                                         label_encoder.transform(label_encoder.classes_)))
# K-Fold Cross-Validation
from sklearn.model_selection import cross_validate
def cross_validation(model, _X, _y, _cv=10):
    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)
      
    return {"Training Accuracy": results['train_accuracy'].mean()*100,
              "Training Precision": results['train_precision'].mean(),
              "Training Recall": results['train_recall'].mean(),
              "Training F1 Score": results['train_f1'].mean(),
              "Test Accuracy": results['test_accuracy'].mean()*100,
              "Test Precision": results['test_precision'].mean(),
              "Test Recall": results['test_recall'].mean(),
              "Test F1 Score": results['test_f1'].mean()
              }
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
MLP_result = cross_validation(MLP, X, encoded_y, 10)
print(MLP_result)


# In[42]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
get_ipython().run_line_magic('matplotlib', 'inline')
X = data3.drop('Class', axis=1)#independent columns
X = (X - X.min())/ (X.max() - X.min())
y =data3['Class']
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y)
label_encoder_name_mapping = dict(zip(label_encoder.classes_,
                                         label_encoder.transform(label_encoder.classes_)))
# K-Fold Cross-Validation
from sklearn.model_selection import cross_validate
def cross_validation(model, _X, _y, _cv=10):
    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)
      
    return {"Mean Training Accuracy": results['train_accuracy'].mean()*100,
              "Mean Training Precision": results['train_precision'].mean(),
              "Mean Training Recall": results['train_recall'].mean(),
              "Mean Training F1 Score": results['train_f1'].mean(),
              "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
              "Mean Validation Precision": results['test_precision'].mean(),
              "Mean Validation Recall": results['test_recall'].mean(),
              "Mean Validation F1 Score": results['test_f1'].mean()
              }
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
DT = DecisionTreeClassifier(max_depth=4)
DT_result = cross_validation(DT, X, encoded_y, 10)
print(DT_result)


# In[43]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.naive_bayes import GaussianNB 
X = data3.drop('Class', axis=1)#independent columns
X = (X - X.min())/ (X.max() - X.min())
y =data3['Class']
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y)
label_encoder_name_mapping = dict(zip(label_encoder.classes_,
                                         label_encoder.transform(label_encoder.classes_)))
# K-Fold Cross-Validation
from sklearn.model_selection import cross_validate
def cross_validation(model, _X, _y, _cv=10):
    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)
      
    return {"Mean Training Accuracy": results['train_accuracy'].mean()*100,
              "Mean Training Precision": results['train_precision'].mean(),
              "Mean Training Recall": results['train_recall'].mean(),
              "Mean Training F1 Score": results['train_f1'].mean(),
              "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
              "Mean Validation Precision": results['test_precision'].mean(),
              "Mean Validation Recall": results['test_recall'].mean(),
              "Mean Validation F1 Score": results['test_f1'].mean()
              }
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
GNB = GaussianNB()
GNB_result = cross_validation(GNB, X, encoded_y, 10)
print(GNB_result)


# In[5]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.svm import SVC
X = data3.drop('Class', axis=1)#independent columns
X = (X - X.min())/ (X.max() - X.min())
y =data3['Class']
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y)
label_encoder_name_mapping = dict(zip(label_encoder.classes_,
                                         label_encoder.transform(label_encoder.classes_)))
# K-Fold Cross-Validation
from sklearn.model_selection import cross_validate
def cross_validation(model, _X, _y, _cv=10):
    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)
      
    return {"Mean Training Accuracy": results['train_accuracy'].mean()*100,
              "Mean Training Precision": results['train_precision'].mean(),
              "Mean Training Recall": results['train_recall'].mean(),
              "Mean Training F1 Score": results['train_f1'].mean(),
              "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
              "Mean Validation Precision": results['test_precision'].mean(),
              "Mean Validation Recall": results['test_recall'].mean(),
              "Mean Validation F1 Score": results['test_f1'].mean()
              }
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

SVM = SVC(kernel = 'linear', C = 1)
SVM_result = cross_validation(SVM, X, encoded_y, 10)
print(SVM_result)


# In[45]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
get_ipython().run_line_magic('matplotlib', 'inline')
X = data3.drop('Class', axis=1)#independent columns
X = (X - X.min())/ (X.max() - X.min())
y =data3['Class']
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y)
label_encoder_name_mapping = dict(zip(label_encoder.classes_,
                                         label_encoder.transform(label_encoder.classes_)))
# K-Fold Cross-Validation
from sklearn.model_selection import cross_validate
def cross_validation(model, _X, _y, _cv=10):
    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)
      
    return {"Mean Training Accuracy": results['train_accuracy'].mean()*100,
              "Mean Training Precision": results['train_precision'].mean(),
              "Mean Training Recall": results['train_recall'].mean(),
              "Mean Training F1 Score": results['train_f1'].mean(),
              "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
              "Mean Validation Precision": results['test_precision'].mean(),
              "Mean Validation Recall": results['test_recall'].mean(),
              "Mean Validation F1 Score": results['test_f1'].mean()
              }
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

KNN = KNeighborsClassifier(n_neighbors = 7)
KNN_result = cross_validation(KNN, X, encoded_y, 10)
print(KNN_result)


# In[ ]:




