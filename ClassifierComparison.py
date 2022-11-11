#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Breast cancer(Without applying Fuzzy gene selection)
#With employing five classifier approaches
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import model_selection
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
data3 = pd.read_csv("C:/Users/Mahmood/Desktop/Camparsion/BreastCancer/Microarray/Microarray/155SamplesBreastCancerSubtypes1.csv")
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
def cross_validation(model, _X, _y, _cv=5):
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
    for i in range (5):
        sum1=sum1+a[i]
        sum2=sum2+b[i]
        sum3=sum3+c[i]
    print("Precision=",sum1/5)
    print("Recall=",sum2/5)
    print("F1-score=",sum3/5)
    
              
def cross_validation1(model, _X, _y, _cv=5):
    d=[]
    sum4=0
    results1 = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring='accuracy',
                               return_train_score=True)
      
    
    return results1['test_score']
              
    # Grouped Bar Chart for both training and validation data
def plot_result(x_label, y_label, plot_title, val_data1,val_data2,val_data3,val_data4,val_data5):
       
        # Set size of plot
        plt.figure(figsize=(10,5))
        labels = ["1", "2", "3", "4","5"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.10000, 1)
        plt.bar(X_axis-0.1, val_data1, 0.1, color='blue', label='DT')
        plt.bar(X_axis-0.2, val_data2, 0.1, color='yellow', label='GNB')
        plt.bar(X_axis-0.3, val_data3, 0.1, color='Pink', label='KNN')
        plt.bar(X_axis-0.4, val_data4, 0.1, color='Purple', label='SVM')
        plt.bar(X_axis-0.5, val_data5, 0.1, color='green', label='MLP')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()
print("Decision Tree Results")
dt=[]
sumdt=0
DT = DecisionTreeClassifier(max_depth=4,random_state=0)
dt=cross_validation1(DT,X,encoded_y,5)
for s in range(5):
    sumdt=sumdt+dt[s]
print("accuracy",sumdt/5) 
cross_validation(DT, X, encoded_y, 5)
print("Gaussian Naive Bayes Results")
gnb=[]
sumgnb=0
GNB =GaussianNB()
gnb=cross_validation1(GNB,X,encoded_y,5)
for s1 in range(5):
    sumgnb=sumgnb+gnb[s1]
print("accuracy",sumgnb/5) 
cross_validation(GNB, X, encoded_y, 5)
print("K-Nearest Neighbors Results")
knn=[]
sumknn=0
KNN =  KNeighborsClassifier(n_neighbors = 7)
knn=cross_validation1(KNN,X,encoded_y,5)
for s2 in range(5):
    sumknn=sumknn+knn[s2]
print("accuracy",sumknn/5) 
cross_validation(KNN, X, encoded_y, 5)
print("Support Vector Machine Results")
svm=[]
sumsvm=0
SVM = SVC(kernel = 'linear', C = 1)
svm=cross_validation1(SVM,X,encoded_y,5)
for s3 in range(5):
    sumsvm=sumsvm+svm[s3]
print("accuracy",sumsvm/5)
cross_validation(SVM, X, encoded_y, 5)
print("Multilayer Perceptron Results")
mpl=[]
sum=0
MLP = MLPClassifier(hidden_layer_sizes=(300,200,100), activation='relu', 
                                max_iter = 200, solver = 'adam',random_state=0)
mlp=cross_validation1(MLP,X,encoded_y,5)
for k in range(5):
    sum=sum+mlp[k]
print("accuracy",sum/5) 
cross_validation(MLP, X, encoded_y, 5)
plot_result("Classifiers",
            "Accuracy",
            "Accuracy scores in 5 Folds",
            dt,gnb,knn,svm,mlp)


# In[2]:


# Breast cancer(Applying Fuzzy gene selection mehtod for selecting small number of genes) 
#With employing five classifier approaches
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import model_selection
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
data4= pd.read_csv("C:/Users/Mahmood/Desktop/Camparsion/BreastCancer/Microarray/FuzzyGeneSelected_Breastcancer/FuzzyGenes.csv")
X = data4.drop('Class', axis=1)#independent columns
y =data4['Class']
scoring = ['precision_macro', 'recall_macro','f1_macro']
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y)
label_encoder_name_mapping = dict(zip(label_encoder.classes_,
                                         label_encoder.transform(label_encoder.classes_)))
# K-Fold Cross-Validation
from sklearn.model_selection import cross_validate
def cross_validation(model, _X, _y, _cv=5):
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
    for i in range (5):
        sum1=sum1+a[i]
        sum2=sum2+b[i]
        sum3=sum3+c[i]
    print("Precision=",sum1/5)
    print("Recall=",sum2/5)
    print("F1-score=",sum3/5)
    
              
def cross_validation1(model, _X, _y, _cv=5):
    d=[]
    sum4=0
    results1 = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring='accuracy',
                               return_train_score=True)
      
    
    return results1['test_score']
              
    # Grouped Bar Chart for both training and validation data
def plot_result(x_label, y_label, plot_title, val_data1,val_data2,val_data3,val_data4,val_data5):
       
        # Set size of plot
        plt.figure(figsize=(10,5))
        labels = ["1", "2", "3", "4","5"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.10000, 1)
        plt.bar(X_axis-0.1, val_data1, 0.1, color='blue', label='DT')
        plt.bar(X_axis-0.2, val_data2, 0.1, color='yellow', label='GNB')
        plt.bar(X_axis-0.3, val_data3, 0.1, color='Pink', label='KNN')
        plt.bar(X_axis-0.4, val_data4, 0.1, color='Purple', label='SVM')
        plt.bar(X_axis-0.5, val_data5, 0.1, color='green', label='MLP')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()
print("Decision Tree Results")
dt=[]
sumdt=0
DT = DecisionTreeClassifier(max_depth=4,random_state=0)
dt=cross_validation1(DT,X,encoded_y,5)
for s in range(5):
    sumdt=sumdt+dt[s]
print("accuracy",sumdt/5) 
cross_validation(DT, X, encoded_y, 5)
print("Gaussian Naive Bayes Results")
gnb=[]
sumgnb=0
GNB =GaussianNB()
gnb=cross_validation1(GNB,X,encoded_y,5)
for s1 in range(5):
    sumgnb=sumgnb+gnb[s1]
print("accuracy",sumgnb/5) 
cross_validation(GNB, X, encoded_y, 5)
print("K-Nearest Neighbors Results")
knn=[]
sumknn=0
KNN =  KNeighborsClassifier(n_neighbors = 7)
knn=cross_validation1(KNN,X,encoded_y,5)
for s2 in range(5):
    sumknn=sumknn+knn[s2]
print("accuracy",sumknn/5) 
cross_validation(KNN, X, encoded_y, 5)
print("Support Vector Machine Results")
svm=[]
sumsvm=0
SVM = SVC(kernel = 'linear', C = 1)
svm=cross_validation1(SVM,X,encoded_y,5)
for s3 in range(5):
    sumsvm=sumsvm+svm[s3]
print("accuracy",sumsvm/5)
cross_validation(SVM, X, encoded_y, 5)
print("Multilayer Perceptron Results")
mpl=[]
sum=0
MLP = MLPClassifier(hidden_layer_sizes=(300,200,100), activation='relu', 
                                max_iter = 200, solver = 'adam',random_state=0)
mlp=cross_validation1(MLP,X,encoded_y,5)
for k in range(5):
    sum=sum+mlp[k]
print("accuracy",sum/5) 
cross_validation(MLP, X, encoded_y, 5)
plot_result("Classifiers",
            "Accuracy",
            "Accuracy scores in 5 Folds",
            dt,gnb,knn,svm,mlp)


# In[3]:


# Breast Five cancer types (Without applying Fuzzy gene selection)
#With employing five classifier approaches
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import model_selection
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
data5 = pd.read_csv("C:/Users/Mahmood/Desktop/Camparsion/fiveTypes/FiveCancerTypes.csv")
X = data5.drop('Class', axis=1)#independent columns
y =data5['Class']
scoring = ['precision_macro', 'recall_macro','f1_macro']
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y)
label_encoder_name_mapping = dict(zip(label_encoder.classes_,
                                         label_encoder.transform(label_encoder.classes_)))
# K-Fold Cross-Validation
from sklearn.model_selection import cross_validate
def cross_validation(model, _X, _y, _cv=5):
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
    for i in range (5):
        sum1=sum1+a[i]
        sum2=sum2+b[i]
        sum3=sum3+c[i]
    print("Precision=",sum1/5)
    print("Recall=",sum2/5)
    print("F1-score=",sum3/5)
    
              
def cross_validation1(model, _X, _y, _cv=5):
    d=[]
    sum4=0
    results1 = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring='accuracy',
                               return_train_score=True)
      
    
    return results1['test_score']
              
    # Grouped Bar Chart for both training and validation data
def plot_result(x_label, y_label, plot_title, val_data1,val_data2,val_data3,val_data4,val_data5):
       
        # Set size of plot
        plt.figure(figsize=(10,5))
        labels = ["1", "2", "3", "4","5"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.10000, 1)
        plt.bar(X_axis-0.1, val_data1, 0.1, color='blue', label='DT')
        plt.bar(X_axis-0.2, val_data2, 0.1, color='yellow', label='GNB')
        plt.bar(X_axis-0.3, val_data3, 0.1, color='Pink', label='KNN')
        plt.bar(X_axis-0.4, val_data4, 0.1, color='Purple', label='SVM')
        plt.bar(X_axis-0.5, val_data5, 0.1, color='green', label='MLP')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()
print("Decision Tree Results")
dt=[]
sumdt=0
DT = DecisionTreeClassifier(max_depth=4,random_state=0)
dt=cross_validation1(DT,X,encoded_y,5)
for s in range(5):
    sumdt=sumdt+dt[s]
print("accuracy",sumdt/5) 
cross_validation(DT, X, encoded_y, 5)
print("Gaussian Naive Bayes Results")
gnb=[]
sumgnb=0
GNB =GaussianNB()
gnb=cross_validation1(GNB,X,encoded_y,5)
for s1 in range(5):
    sumgnb=sumgnb+gnb[s1]
print("accuracy",sumgnb/5) 
cross_validation(GNB, X, encoded_y, 5)
print("K-Nearest Neighbors Results")
knn=[]
sumknn=0
KNN =  KNeighborsClassifier(n_neighbors = 7)
knn=cross_validation1(KNN,X,encoded_y,5)
for s2 in range(5):
    sumknn=sumknn+knn[s2]
print("accuracy",sumknn/5) 
cross_validation(KNN, X, encoded_y, 5)
print("Support Vector Machine Results")
svm=[]
sumsvm=0
SVM = SVC(kernel = 'linear', C = 1)
svm=cross_validation1(SVM,X,encoded_y,5)
for s3 in range(5):
    sumsvm=sumsvm+svm[s3]
print("accuracy",sumsvm/5)
cross_validation(SVM, X, encoded_y, 5)
print("Multilayer Perceptron Results")
mpl=[]
sum=0
MLP = MLPClassifier(hidden_layer_sizes=(300,200,100), activation='relu', 
                                max_iter = 200, solver = 'adam',random_state=0)
mlp=cross_validation1(MLP,X,encoded_y,5)
for k in range(5):
    sum=sum+mlp[k]
print("accuracy",sum/5) 
cross_validation(MLP, X, encoded_y, 5)
plot_result("Classifiers",
            "Accuracy",
            "Accuracy scores in 5 Folds",
            dt,gnb,knn,svm,mlp)


# In[16]:


# Five cancer types (applying Fuzzy gene selection)
#With employing five classifier approaches
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import model_selection
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
data6= pd.read_csv("C:/Users/Mahmood/Desktop/Camparsion/fiveTypes/FiveTypes_Fuzzy/FuzzyGenes.csv")
X = data6.drop('Class', axis=1)#independent columns
X = (X - X.min())/ (X.max() - X.min())
y =data6['Class']
scoring = ['precision_macro', 'recall_macro','f1_macro']
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y)
label_encoder_name_mapping = dict(zip(label_encoder.classes_,
                                         label_encoder.transform(label_encoder.classes_)))
# K-Fold Cross-Validation
from sklearn.model_selection import cross_validate
def cross_validation(model, _X, _y, _cv=5):
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
    for i in range (5):
        sum1=sum1+a[i]
        sum2=sum2+b[i]
        sum3=sum3+c[i]
    print("Precision=",sum1/5)
    print("Recall=",sum2/5)
    print("F1-score=",sum3/5)
    
              
def cross_validation1(model, _X, _y, _cv=5):
    d=[]
    sum4=0
    results1 = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring='accuracy',
                               return_train_score=True)
      
    
    return results1['test_score']
              
    # Grouped Bar Chart for both training and validation data
def plot_result(x_label, y_label, plot_title, val_data1,val_data2,val_data3,val_data4,val_data5):
       
        # Set size of plot
        plt.figure(figsize=(10,5))
        labels = ["1", "2", "3", "4","5"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.10000, 1)
        plt.bar(X_axis-0.1, val_data1, 0.1, color='blue', label='DT')
        plt.bar(X_axis-0.2, val_data2, 0.1, color='yellow', label='GNB')
        plt.bar(X_axis-0.3, val_data3, 0.1, color='Pink', label='KNN')
        plt.bar(X_axis-0.4, val_data4, 0.1, color='Purple', label='SVM')
        plt.bar(X_axis-0.5, val_data5, 0.1, color='green', label='MLP')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()
print("Decision Tree Results")
dt=[]
sumdt=0
DT = DecisionTreeClassifier(max_depth=4,random_state=0)
dt=cross_validation1(DT,X,encoded_y,5)
for s in range(5):
    sumdt=sumdt+dt[s]
print("accuracy",sumdt/5) 
cross_validation(DT, X, encoded_y, 5)
print("Gaussian Naive Bayes Results")
gnb=[]
sumgnb=0
GNB =GaussianNB()
gnb=cross_validation1(GNB,X,encoded_y,5)
for s1 in range(5):
    sumgnb=sumgnb+gnb[s1]
print("accuracy",sumgnb/5) 
cross_validation(GNB, X, encoded_y, 5)
print("K-Nearest Neighbors Results")
knn=[]
sumknn=0
KNN =  KNeighborsClassifier(n_neighbors = 7)
knn=cross_validation1(KNN,X,encoded_y,5)
for s2 in range(5):
    sumknn=sumknn+knn[s2]
print("accuracy",sumknn/5) 
cross_validation(KNN, X, encoded_y, 5)
print("Support Vector Machine Results")
svm=[]
sumsvm=0
SVM = SVC(kernel = 'linear', C = 1)
svm=cross_validation1(SVM,X,encoded_y,5)
for s3 in range(5):
    sumsvm=sumsvm+svm[s3]
print("accuracy",sumsvm/5)
cross_validation(SVM, X, encoded_y, 5)
print("Multilayer Perceptron Results")
mpl=[]
sum=0
MLP = MLPClassifier(hidden_layer_sizes=(300,200,100), activation='relu', 
                                max_iter = 200, solver = 'adam',random_state=0)
mlp=cross_validation1(MLP,X,encoded_y,5)
for k in range(5):
    sum=sum+mlp[k]
print("accuracy",sum/5) 
cross_validation(MLP, X, encoded_y, 5)
plot_result("Classifiers",
            "Accuracy",
            "Accuracy scores in 5 Folds",
            dt,gnb,knn,svm,mlp)


# In[5]:


# Thyroid cancer (without applying Fuzzy gene selection)
#With employing five classifier approaches
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import model_selection
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
data7 = pd.read_csv("C:/Users/Mahmood/Desktop/Camparsion/thyroid/Microarray/Orginalthyroid.csv")
data7 = data7.iloc[: , 1:]
X = data7.drop('Class', axis=1)#independent columns
X = (X - X.min())/ (X.max() - X.min())
y =data7['Class']
scoring = ['precision_macro', 'recall_macro','f1_macro']
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y)
label_encoder_name_mapping = dict(zip(label_encoder.classes_,
                                         label_encoder.transform(label_encoder.classes_)))
# K-Fold Cross-Validation
from sklearn.model_selection import cross_validate
def cross_validation(model, _X, _y, _cv=5):
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
    for i in range (5):
        sum1=sum1+a[i]
        sum2=sum2+b[i]
        sum3=sum3+c[i]
    print("Precision=",sum1/5)
    print("Recall=",sum2/5)
    print("F1-score=",sum3/5)
    
              
def cross_validation1(model, _X, _y, _cv=5):
    d=[]
    sum4=0
    results1 = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring='accuracy',
                               return_train_score=True)
      
    
    return results1['test_score']
              
    # Grouped Bar Chart for both training and validation data
def plot_result(x_label, y_label, plot_title, val_data1,val_data2,val_data3,val_data4,val_data5):
       
        # Set size of plot
        plt.figure(figsize=(10,5))
        labels = ["1", "2", "3", "4","5"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.10000, 1)
        plt.bar(X_axis-0.1, val_data1, 0.1, color='blue', label='DT')
        plt.bar(X_axis-0.2, val_data2, 0.1, color='yellow', label='GNB')
        plt.bar(X_axis-0.3, val_data3, 0.1, color='Pink', label='KNN')
        plt.bar(X_axis-0.4, val_data4, 0.1, color='Purple', label='SVM')
        plt.bar(X_axis-0.5, val_data5, 0.1, color='green', label='MLP')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()
print("Decision Tree Results")
dt=[]
sumdt=0
DT = DecisionTreeClassifier(max_depth=4,random_state=0)
dt=cross_validation1(DT,X,encoded_y,5)
for s in range(5):
    sumdt=sumdt+dt[s]
print("accuracy",sumdt/5) 
cross_validation(DT, X, encoded_y, 5)
print("Gaussian Naive Bayes Results")
gnb=[]
sumgnb=0
GNB =GaussianNB()
gnb=cross_validation1(GNB,X,encoded_y,5)
for s1 in range(5):
    sumgnb=sumgnb+gnb[s1]
print("accuracy",sumgnb/5) 
cross_validation(GNB, X, encoded_y, 5)
print("K-Nearest Neighbors Results")
knn=[]
sumknn=0
KNN =  KNeighborsClassifier(n_neighbors = 7)
knn=cross_validation1(KNN,X,encoded_y,5)
for s2 in range(5):
    sumknn=sumknn+knn[s2]
print("accuracy",sumknn/5) 
cross_validation(KNN, X, encoded_y, 5)
print("Support Vector Machine Results")
svm=[]
sumsvm=0
SVM = SVC(kernel = 'linear', C = 1)
svm=cross_validation1(SVM,X,encoded_y,5)
for s3 in range(5):
    sumsvm=sumsvm+svm[s3]
print("accuracy",sumsvm/5)
cross_validation(SVM, X, encoded_y, 5)
print("Multilayer Perceptron Results")
mpl=[]
sum=0
MLP = MLPClassifier(hidden_layer_sizes=(300,200,100), activation='relu', 
                                max_iter = 200, solver = 'adam',random_state=0)
mlp=cross_validation1(MLP,X,encoded_y,5)
for k in range(5):
    sum=sum+mlp[k]
print("accuracy",sum/5) 
cross_validation(MLP, X, encoded_y, 5)
plot_result("Classifiers",
            "Accuracy",
            "Accuracy scores in 5 Folds",
            dt,gnb,knn,svm,mlp)


# In[4]:


# Thyroid cancer (applying Fuzzy gene selection)
#With employing five classifier approaches
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import model_selection
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
data8 = pd.read_csv("C:/Users/Mahmood/Desktop/Camparsion/thyroid/Microarray/FuzzyGenes.csv")
X = data8.drop('Class', axis=1)#independent columns
X = (X - X.min())/ (X.max() - X.min())
y =data8['Class']
scoring = ['precision_macro', 'recall_macro','f1_macro']
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y)
label_encoder_name_mapping = dict(zip(label_encoder.classes_,
                                         label_encoder.transform(label_encoder.classes_)))
# K-Fold Cross-Validation
from sklearn.model_selection import cross_validate
def cross_validation(model, _X, _y, _cv=5):
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
    for i in range (5):
        sum1=sum1+a[i]
        sum2=sum2+b[i]
        sum3=sum3+c[i]
    print("Precision=",sum1/5)
    print("Recall=",sum2/5)
    print("F1-score=",sum3/5)
    
              
def cross_validation1(model, _X, _y, _cv=5):
    d=[]
    sum4=0
    results1 = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring='accuracy',
                               return_train_score=True)
      
    
    return results1['test_score']
              
    # Grouped Bar Chart for both training and validation data
def plot_result(x_label, y_label, plot_title, val_data1,val_data2,val_data3,val_data4,val_data5):
       
        # Set size of plot
        plt.figure(figsize=(10,5))
        labels = ["1", "2", "3", "4","5"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.20000, 1)
        plt.bar(X_axis-0.1, val_data1, 0.1, color='blue', label='DT')
        plt.bar(X_axis-0.2, val_data2, 0.1, color='yellow', label='GNB')
        plt.bar(X_axis-0.3, val_data3, 0.1, color='Pink', label='KNN')
        plt.bar(X_axis-0.4, val_data4, 0.1, color='Purple', label='SVM')
        plt.bar(X_axis-0.5, val_data5, 0.1, color='green', label='MLP')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()
print("Decision Tree Results")
dt=[]
sumdt=0
DT = DecisionTreeClassifier(max_depth=4,random_state=0)
dt=cross_validation1(DT,X,encoded_y,5)
for s in range(5):
    sumdt=sumdt+dt[s]
print("accuracy",sumdt/5) 
cross_validation(DT, X, encoded_y, 5)
print("Gaussian Naive Bayes Results")
gnb=[]
sumgnb=0
GNB =GaussianNB()
gnb=cross_validation1(GNB,X,encoded_y,5)
for s1 in range(5):
    sumgnb=sumgnb+gnb[s1]
print("accuracy",sumgnb/5) 
cross_validation(GNB, X, encoded_y, 5)
print("K-Nearest Neighbors Results")
knn=[]
sumknn=0
KNN =  KNeighborsClassifier(n_neighbors = 7)
knn=cross_validation1(KNN,X,encoded_y,5)
for s2 in range(5):
    sumknn=sumknn+knn[s2]
print("accuracy",sumknn/5) 
cross_validation(KNN, X, encoded_y, 5)
print("Support Vector Machine Results")
svm=[]
sumsvm=0
SVM = SVC(kernel = 'linear', C = 1)
svm=cross_validation1(SVM,X,encoded_y,5)
for s3 in range(5):
    sumsvm=sumsvm+svm[s3]
print("accuracy",sumsvm/5)
cross_validation(SVM, X, encoded_y, 5)
print("Multilayer Perceptron Results")
mpl=[]
sum=0
MLP = MLPClassifier(hidden_layer_sizes=(300,200,100), activation='relu', 
                                max_iter = 200, solver = 'adam')
mlp=cross_validation1(MLP,X,encoded_y,5)
for k in range(5):
    sum=sum+mlp[k]
print("accuracy",sum/5) 
cross_validation(MLP, X, encoded_y, 5)
plot_result("Classifiers",
            "Accuracy",
            "Accuracy scores in 5 Folds",
            dt,gnb,knn,svm,mlp)


# In[9]:


# Liver cancer(GSE14520) without applying fuzzy gene selection method, employing five classifier techniques
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import model_selection
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
data9= pd.read_csv("C:/Users/Mahmood/Desktop/Camparsion/LiverCancer/Microarray/GSE14520-GPL3921_series_matrix.txt/StructuredData.csv") 
data9 = data9.drop('Samples', axis=1)
X = data9.drop('Class', axis=1)#independent columns
y =data9['Class']
label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y)
label_encoder_name_mapping = dict(zip(label_encoder.classes_,
                                         label_encoder.transform(label_encoder.classes_)))
# K-Fold Cross-Validation
from sklearn.model_selection import cross_validate
def cross_validation(model, _X, _y, _cv=5):
    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)
      
    return {"Validation Accuracy scores": results['test_accuracy'],
              "Accuracy": results['test_accuracy'].mean()*100,
              "Precision": results['test_precision'].mean(),
              "Recall": results['test_recall'].mean(),
              "F1-Score": results['test_f1'].mean()
              }
    # Grouped Bar Chart for both training and validation data
def plot_result(x_label, y_label, plot_title, val_data1,val_data2,val_data3,val_data4,val_data5):
       
        # Set size of plot
        plt.figure(figsize=(10,5))
        labels = ["1", "2", "3", "4","5"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.10000, 1)
        plt.bar(X_axis-0.1, val_data1, 0.1, color='blue', label='DT')
        plt.bar(X_axis-0.2, val_data2, 0.1, color='yellow', label='GNB')
        plt.bar(X_axis-0.3, val_data3, 0.1, color='Pink', label='KNN')
        plt.bar(X_axis-0.4, val_data4, 0.1, color='Purple', label='SVM')
        plt.bar(X_axis-0.5, val_data5, 0.1, color='green', label='MLP')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()
DT = DecisionTreeClassifier(max_depth=4)
DT_result = cross_validation(DT, X, encoded_y, 5)
print("** Decision Tree Results **")
print("Accuracy=",DT_result["Accuracy"])
print("Precision=",DT_result["Precision"])
print("Recall=",DT_result["Recall"])
print("F1-score=",DT_result["F1-Score"])
GNB = GaussianNB()
GNB_result = cross_validation(GNB, X, encoded_y, 5)
print("** Gaussian Naive Bayes Results **")
print("Accuracy=",GNB_result["Accuracy"])
print("Precision=",GNB_result["Precision"])
print("Recall=",GNB_result["Recall"])
print("F1-score=",GNB_result["F1-Score"])
KNN = KNeighborsClassifier(n_neighbors = 7)
KNN_result = cross_validation(KNN, X, encoded_y, 5)
print("** K-Nearest Neighbors Results* *")
print("Accuracy=",KNN_result["Accuracy"])
print("Precision=",KNN_result["Precision"])
print("Recall=",KNN_result["Recall"])
print("F1-score=",KNN_result["F1-Score"])
SVM = SVC(kernel = 'linear', C = 1)
SVM_result = cross_validation(SVM, X, encoded_y, 5)
print("** Support Vector Machine Results **")
print("Accuracy=",SVM_result["Accuracy"])
print("Precision=",SVM_result["Precision"])
print("Recall=",SVM_result["Recall"])
print("F1-score=",SVM_result["F1-Score"])
MLP = MLPClassifier(hidden_layer_sizes=(300,200,100), activation='relu', 
                                max_iter = 200, solver = 'adam',random_state=0)
MLP_result = cross_validation(MLP, X, encoded_y, 5)
print("** Multilayer Perceptron Results **")
print("Accuracy=",MLP_result["Accuracy"])
print("Precision=",MLP_result["Precision"])
print("Recall=",MLP_result["Recall"])
print("F1-score=",MLP_result["F1-Score"])
plot_result("Classifier Techniques",
            "Accuracy",
            "Accuracy scores in 5 Folds",
            DT_result["Validation Accuracy scores"],GNB_result["Validation Accuracy scores"], KNN_result["Validation Accuracy scores"],SVM_result["Validation Accuracy scores"],MLP_result["Validation Accuracy scores"])


# In[12]:


# Liver cancer(GSE14520) and  applying fuzzy gene selection method, employing five classifier techniques
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import model_selection
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
data8= pd.read_csv("C:/Users/Mahmood/Desktop/Camparsion/LiverCancer/Microarray/LiverCancer_GSE14520/FuzzyGenes.csv")
X = data8.drop('Class', axis=1)#independent columns
y =data8['Class']
label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y)
label_encoder_name_mapping = dict(zip(label_encoder.classes_,
                                         label_encoder.transform(label_encoder.classes_)))
# K-Fold Cross-Validation
from sklearn.model_selection import cross_validate
def cross_validation(model, _X, _y, _cv=5):
    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)
      
    return {"Validation Accuracy scores": results['test_accuracy'],
              "Accuracy": results['test_accuracy'].mean()*100,
              "Precision": results['test_precision'].mean(),
              "Recall": results['test_recall'].mean(),
              "F1-Score": results['test_f1'].mean()
              }
    # Grouped Bar Chart for both training and validation data
def plot_result(x_label, y_label, plot_title, val_data1,val_data2,val_data3,val_data4,val_data5):
       
        # Set size of plot
        plt.figure(figsize=(10,5))
        labels = ["1", "2", "3", "4","5"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.10000, 1)
        plt.bar(X_axis-0.1, val_data1, 0.1, color='blue', label='DT')
        plt.bar(X_axis-0.2, val_data2, 0.1, color='yellow', label='GNB')
        plt.bar(X_axis-0.3, val_data3, 0.1, color='Pink', label='KNN')
        plt.bar(X_axis-0.4, val_data4, 0.1, color='Purple', label='SVM')
        plt.bar(X_axis-0.5, val_data5, 0.1, color='green', label='MLP')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()
DT = DecisionTreeClassifier(max_depth=4)
DT_result = cross_validation(DT, X, encoded_y, 5)
print("** Decision Tree Results **")
print("Accuracy=",DT_result["Accuracy"])
print("Precision=",DT_result["Precision"])
print("Recall=",DT_result["Recall"],"=")
print("F1-score=",DT_result["F1-Score"])
GNB = GaussianNB()
GNB_result = cross_validation(GNB, X, encoded_y, 5)
print("** Gaussian Naive Bayes Results **")
print("Accuracy=",GNB_result["Accuracy"])
print("Precision=",GNB_result["Precision"])
print("Recall=",GNB_result["Recall"])
print("F1-score=",GNB_result["F1-Score"])
KNN = KNeighborsClassifier(n_neighbors = 7)
KNN_result = cross_validation(KNN, X, encoded_y, 5)
print("** K-Nearest Neighbors Results **")
print("Accuracy=",KNN_result["Accuracy"])
print("Precision=",KNN_result["Precision"])
print("Recall=",KNN_result["Recall"])
print("F1-score=",KNN_result["F1-Score"])
SVM = SVC(kernel = 'linear', C = 1)
SVM_result = cross_validation(SVM, X, encoded_y, 5)
print("** Support Vector Machine Results **")
print("Accuracy=",SVM_result["Accuracy"])
print("Precision=",SVM_result["Precision"])
print("Recall=",SVM_result["Recall"])
print("F1-score=",SVM_result["F1-Score"])
MLP = MLPClassifier(hidden_layer_sizes=(300,200,100), activation='relu', 
                                max_iter = 200, solver = 'adam')
MLP_result = cross_validation(MLP, X, encoded_y, 5)
print("** Multilayer Perceptron Results **")
print("Accuracy=",MLP_result["Accuracy"])
print("Precision=",MLP_result["Precision"])
print("Recall=",MLP_result["Recall"])
print("F1-score=",MLP_result["F1-Score"])
plot_result("Classifier Techniques",
            "Accuracy",
            "Accuracy scores in 5 Folds",
            DT_result["Validation Accuracy scores"],GNB_result["Validation Accuracy scores"], KNN_result["Validation Accuracy scores"],SVM_result["Validation Accuracy scores"],MLP_result["Validation Accuracy scores"])


# In[15]:


# Liver cancer(GSE77314) without applying fuzzy gene selection method, employing five classifier techniques
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import model_selection
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
data10 = pd.read_csv("C:/Users/Mahmood/Desktop/Camparsion/LiverCancer/RNA-seq/Deep Learning/StructureLiverCancer1.csv")
X = data10.drop('Class', axis=1)#independent columns
y =data10['Class']
label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y)
label_encoder_name_mapping = dict(zip(label_encoder.classes_,
                                         label_encoder.transform(label_encoder.classes_)))
# K-Fold Cross-Validation
from sklearn.model_selection import cross_validate
def cross_validation(model, _X, _y, _cv=5):
    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)
      
    return {"Validation Accuracy scores": results['test_accuracy'],
              "Accuracy": results['test_accuracy'].mean()*100,
              "Precision": results['test_precision'].mean(),
              "Recall": results['test_recall'].mean(),
              "F1-Score": results['test_f1'].mean()
              }
    # Grouped Bar Chart for both training and validation data
def plot_result(x_label, y_label, plot_title, val_data1,val_data2,val_data3,val_data4,val_data5):
       
        # Set size of plot
        plt.figure(figsize=(10,5))
        labels = ["1", "2", "3", "4","5"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.10000, 1)
        plt.bar(X_axis-0.1, val_data1, 0.1, color='blue', label='DT')
        plt.bar(X_axis-0.2, val_data2, 0.1, color='yellow', label='GNB')
        plt.bar(X_axis-0.3, val_data3, 0.1, color='Pink', label='KNN')
        plt.bar(X_axis-0.4, val_data4, 0.1, color='Purple', label='SVM')
        plt.bar(X_axis-0.5, val_data5, 0.1, color='green', label='MLP')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()
DT = DecisionTreeClassifier(max_depth=4)
DT_result = cross_validation(DT, X, encoded_y, 5)
print("** Decision Tree Results **")
print("Accuracy=",DT_result["Accuracy"])
print("Precision=",DT_result["Precision"])
print("Recall=",DT_result["Recall"])
print("F1-score=",DT_result["F1-Score"])
GNB = GaussianNB()
GNB_result = cross_validation(GNB, X, encoded_y, 5)
print("** Gaussian Naive Bayes Results **")
print("Accuracy=",GNB_result["Accuracy"])
print("Precision=",GNB_result["Precision"])
print("Recall=",GNB_result["Recall"])
print("F1-score=",GNB_result["F1-Score"])
KNN = KNeighborsClassifier(n_neighbors = 7)
KNN_result = cross_validation(KNN, X, encoded_y, 5)
print("** K-Nearest Neighbors Results* *")
print("Accuracy=",KNN_result["Accuracy"])
print("Precision=",KNN_result["Precision"])
print("Recall=",KNN_result["Recall"])
print("F1-score=",KNN_result["F1-Score"])
SVM = SVC(kernel = 'linear', C = 1)
SVM_result = cross_validation(SVM, X, encoded_y, 5)
print("** Support Vector Machine Results **")
print("Accuracy=",SVM_result["Accuracy"])
print("Precision=",SVM_result["Precision"])
print("Recall=",SVM_result["Recall"])
print("F1-score=",SVM_result["F1-Score"])
MLP = MLPClassifier(hidden_layer_sizes=(300,200,100), activation='relu', 
                                max_iter = 200, solver = 'adam',random_state=0)
MLP_result = cross_validation(MLP, X, encoded_y, 5)
print("** Multilayer Perceptron Results **")
print("Accuracy=",MLP_result["Accuracy"])
print("Precision=",MLP_result["Precision"])
print("Recall=",MLP_result["Recall"])
print("F1-score=",MLP_result["F1-Score"])
plot_result("Classifier Techniques",
            "Accuracy",
            "Accuracy scores in 5 Folds",
            DT_result["Validation Accuracy scores"],GNB_result["Validation Accuracy scores"], KNN_result["Validation Accuracy scores"],SVM_result["Validation Accuracy scores"],MLP_result["Validation Accuracy scores"])


# In[17]:


# Liver cancer(GSE77314) and applying fuzzy gene selection method, employing five classifier techniques
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import model_selection
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
data11 = pd.read_csv("C:/Users/Mahmood/Desktop/Camparsion/LiverCancer/RNA-seq/LiverCancer_GSE77314/FuzzyGenes.csv")
X = data11.drop('Class', axis=1)#independent columns
X = (X - X.min())/ (X.max() - X.min())
y =data11['Class']
label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y)
label_encoder_name_mapping = dict(zip(label_encoder.classes_,
                                         label_encoder.transform(label_encoder.classes_)))
# K-Fold Cross-Validation
from sklearn.model_selection import cross_validate
def cross_validation(model, _X, _y, _cv=5):
    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)
      
    return {"Validation Accuracy scores": results['test_accuracy'],
              "Accuracy": results['test_accuracy'].mean()*100,
              "Precision": results['test_precision'].mean(),
              "Recall": results['test_recall'].mean(),
              "F1-Score": results['test_f1'].mean()
              }
    # Grouped Bar Chart for both training and validation data
def plot_result(x_label, y_label, plot_title, val_data1,val_data2,val_data3,val_data4,val_data5):
       
        # Set size of plot
        plt.figure(figsize=(10,5))
        labels = ["1", "2", "3", "4","5"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.10000, 1)
        plt.bar(X_axis-0.1, val_data1, 0.1, color='blue', label='DT')
        plt.bar(X_axis-0.2, val_data2, 0.1, color='yellow', label='GNB')
        plt.bar(X_axis-0.3, val_data3, 0.1, color='Pink', label='KNN')
        plt.bar(X_axis-0.4, val_data4, 0.1, color='Purple', label='SVM')
        plt.bar(X_axis-0.5, val_data5, 0.1, color='green', label='MLP')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()
DT = DecisionTreeClassifier(max_depth=4)
DT_result = cross_validation(DT, X, encoded_y, 5)
print("** Decision Tree Results **")
print("Accuracy=",DT_result["Accuracy"])
print("Precision=",DT_result["Precision"])
print("Recall=",DT_result["Recall"])
print("F1-score=",DT_result["F1-Score"])
GNB = GaussianNB()
GNB_result = cross_validation(GNB, X, encoded_y, 5)
print("** Gaussian Naive Bayes Results **")
print("Accuracy=",GNB_result["Accuracy"])
print("Precision=",GNB_result["Precision"])
print("Recall=",GNB_result["Recall"])
print("F1-score=",GNB_result["F1-Score"])
KNN = KNeighborsClassifier(n_neighbors = 7)
KNN_result = cross_validation(KNN, X, encoded_y, 5)
print("** K-Nearest Neighbors Results* *")
print("Accuracy=",KNN_result["Accuracy"])
print("Precision=",KNN_result["Precision"])
print("Recall=",KNN_result["Recall"])
print("F1-score=",KNN_result["F1-Score"])
SVM = SVC(kernel = 'linear', C = 1)
SVM_result = cross_validation(SVM, X, encoded_y, 5)
print("** Support Vector Machine Results **")
print("Accuracy=",SVM_result["Accuracy"])
print("Precision=",SVM_result["Precision"])
print("Recall=",SVM_result["Recall"])
print("F1-score=",SVM_result["F1-Score"])
MLP = MLPClassifier(hidden_layer_sizes=(300,200,100), activation='relu', 
                                max_iter = 200, solver = 'adam',random_state=0)
MLP_result = cross_validation(MLP, X, encoded_y, 5)
print("** Multilayer Perceptron Results **")
print("Accuracy=",MLP_result["Accuracy"])
print("Precision=",MLP_result["Precision"])
print("Recall=",MLP_result["Recall"])
print("F1-score=",MLP_result["F1-Score"])
plot_result("Classifier Techniques",
            "Accuracy",
            "Accuracy scores in 5 Folds",
            DT_result["Validation Accuracy scores"],GNB_result["Validation Accuracy scores"], KNN_result["Validation Accuracy scores"],SVM_result["Validation Accuracy scores"],MLP_result["Validation Accuracy scores"])


# In[21]:


# Lung cancer(GSE19804) without applying fuzzy gene selection method, employing five classifier techniques
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import model_selection
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
data12 = pd.read_csv("C:/Users/Mahmood/Desktop/Camparsion/LungCancer/Microarray/OrginalData/GSE19804/Preprossing/DataStructured3.csv")
X = data12.drop('Class', axis=1)#independent columns
y =data12['Class']
label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y)
label_encoder_name_mapping = dict(zip(label_encoder.classes_,
                                         label_encoder.transform(label_encoder.classes_)))
# K-Fold Cross-Validation
from sklearn.model_selection import cross_validate
def cross_validation(model, _X, _y, _cv=5):
    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)
      
    return {"Validation Accuracy scores": results['test_accuracy'],
              "Accuracy": results['test_accuracy'].mean()*100,
              "Precision": results['test_precision'].mean(),
              "Recall": results['test_recall'].mean(),
              "F1-Score": results['test_f1'].mean()
              }
    # Grouped Bar Chart for both training and validation data
def plot_result(x_label, y_label, plot_title, val_data1,val_data2,val_data3,val_data4,val_data5):
       
        # Set size of plot
        plt.figure(figsize=(10,5))
        labels = ["1", "2", "3", "4","5"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.10000, 1)
        plt.bar(X_axis-0.1, val_data1, 0.1, color='blue', label='DT')
        plt.bar(X_axis-0.2, val_data2, 0.1, color='yellow', label='GNB')
        plt.bar(X_axis-0.3, val_data3, 0.1, color='Pink', label='KNN')
        plt.bar(X_axis-0.4, val_data4, 0.1, color='Purple', label='SVM')
        plt.bar(X_axis-0.5, val_data5, 0.1, color='green', label='MLP')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()
DT = DecisionTreeClassifier(max_depth=4)
DT_result = cross_validation(DT, X, encoded_y, 5)
print("** Decision Tree Results **")
print("Accuracy=",DT_result["Accuracy"])
print("Precision=",DT_result["Precision"])
print("Recall=",DT_result["Recall"])
print("F1-score=",DT_result["F1-Score"])
GNB = GaussianNB()
GNB_result = cross_validation(GNB, X, encoded_y, 5)
print("** Gaussian Naive Bayes Results **")
print("Accuracy=",GNB_result["Accuracy"])
print("Precision=",GNB_result["Precision"])
print("Recall=",GNB_result["Recall"])
print("F1-score=",GNB_result["F1-Score"])
KNN = KNeighborsClassifier(n_neighbors = 7)
KNN_result = cross_validation(KNN, X, encoded_y, 5)
print("** K-Nearest Neighbors Results* *")
print("Accuracy=",KNN_result["Accuracy"])
print("Precision=",KNN_result["Precision"])
print("Recall=",KNN_result["Recall"])
print("F1-score=",KNN_result["F1-Score"])
SVM = SVC(kernel = 'linear', C = 1)
SVM_result = cross_validation(SVM, X, encoded_y, 5)
print("** Support Vector Machine Results **")
print("Accuracy=",SVM_result["Accuracy"])
print("Precision=",SVM_result["Precision"])
print("Recall=",SVM_result["Recall"])
print("F1-score=",SVM_result["F1-Score"])
MLP = MLPClassifier(hidden_layer_sizes=(300,200,100), activation='relu', 
                                max_iter = 200, solver = 'adam',random_state=0)
MLP_result = cross_validation(MLP, X, encoded_y, 5)
print("** Multilayer Perceptron Results **")
print("Accuracy=",MLP_result["Accuracy"])
print("Precision=",MLP_result["Precision"])
print("Recall=",MLP_result["Recall"])
print("F1-score=",MLP_result["F1-Score"])
plot_result("Classifier Techniques",
            "Accuracy",
            "Accuracy scores in 5 Folds",
            DT_result["Validation Accuracy scores"],GNB_result["Validation Accuracy scores"], KNN_result["Validation Accuracy scores"],SVM_result["Validation Accuracy scores"],MLP_result["Validation Accuracy scores"])


# In[19]:


# Lung cancer(GSE19804) and applying fuzzy gene selection method, employing five classifier techniques
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import model_selection
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
data13= pd.read_csv("C:/Users/Mahmood/Desktop/Camparsion/LungCancer/Microarray/DataSetOfLungCancer/GSE19804/SelectedGenes.csv")
X = data13.drop('Class', axis=1)#independent columns
X = (X - X.min())/ (X.max() - X.min())
y =data13['Class']
label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y)
label_encoder_name_mapping = dict(zip(label_encoder.classes_,
                                         label_encoder.transform(label_encoder.classes_)))
# K-Fold Cross-Validation
from sklearn.model_selection import cross_validate
def cross_validation(model, _X, _y, _cv=5):
    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model,
                               X=_X,
                               y=_y,
                               cv=_cv,
                               scoring=_scoring,
                               return_train_score=True)
      
    return {"Validation Accuracy scores": results['test_accuracy'],
              "Accuracy": results['test_accuracy'].mean()*100,
              "Precision": results['test_precision'].mean(),
              "Recall": results['test_recall'].mean(),
              "F1-Score": results['test_f1'].mean()
              }
    # Grouped Bar Chart for both training and validation data
def plot_result(x_label, y_label, plot_title, val_data1,val_data2,val_data3,val_data4,val_data5):
       
        # Set size of plot
        plt.figure(figsize=(10,5))
        labels = ["1", "2", "3", "4","5"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.10000, 1)
        plt.bar(X_axis-0.1, val_data1, 0.1, color='blue', label='DT')
        plt.bar(X_axis-0.2, val_data2, 0.1, color='yellow', label='GNB')
        plt.bar(X_axis-0.3, val_data3, 0.1, color='Pink', label='KNN')
        plt.bar(X_axis-0.4, val_data4, 0.1, color='Purple', label='SVM')
        plt.bar(X_axis-0.5, val_data5, 0.1, color='green', label='MLP')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()
DT = DecisionTreeClassifier(max_depth=4)
DT_result = cross_validation(DT, X, encoded_y, 5)
print("** Decision Tree Results **")
print("Accuracy=",DT_result["Accuracy"])
print("Precision=",DT_result["Precision"])
print("Recall=",DT_result["Recall"])
print("F1-score=",DT_result["F1-Score"])
GNB = GaussianNB()
GNB_result = cross_validation(GNB, X, encoded_y, 5)
print("** Gaussian Naive Bayes Results **")
print("Accuracy=",GNB_result["Accuracy"])
print("Precision=",GNB_result["Precision"])
print("Recall=",GNB_result["Recall"])
print("F1-score=",GNB_result["F1-Score"])
KNN = KNeighborsClassifier(n_neighbors = 7)
KNN_result = cross_validation(KNN, X, encoded_y, 5)
print("** K-Nearest Neighbors Results* *")
print("Accuracy=",KNN_result["Accuracy"])
print("Precision=",KNN_result["Precision"])
print("Recall=",KNN_result["Recall"])
print("F1-score=",KNN_result["F1-Score"])
SVM = SVC(kernel = 'linear', C = 1)
SVM_result = cross_validation(SVM, X, encoded_y, 5)
print("** Support Vector Machine Results **")
print("Accuracy=",SVM_result["Accuracy"])
print("Precision=",SVM_result["Precision"])
print("Recall=",SVM_result["Recall"])
print("F1-score=",SVM_result["F1-Score"])
MLP = MLPClassifier(hidden_layer_sizes=(300,200,100), activation='relu', 
                                max_iter = 200, solver = 'adam',random_state=0)
MLP_result = cross_validation(MLP, X, encoded_y, 5)
print("** Multilayer Perceptron Results **")
print("Accuracy=",MLP_result["Accuracy"])
print("Precision=",MLP_result["Precision"])
print("Recall=",MLP_result["Recall"])
print("F1-score=",MLP_result["F1-Score"])
plot_result("Classifier Techniques",
            "Accuracy",
            "Accuracy scores in 5 Folds",
            DT_result["Validation Accuracy scores"],GNB_result["Validation Accuracy scores"], KNN_result["Validation Accuracy scores"],SVM_result["Validation Accuracy scores"],MLP_result["Validation Accuracy scores"])


# In[ ]:




