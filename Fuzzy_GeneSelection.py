#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
data2 = pd.read_csv("C:/Users/Mahmood/Desktop/Camparsion/LungCancer/Microarray/OrginalData/GSE19804/Preprossing/DataStructured3.csv")


# In[1]:


# First genes selection method (mutual information)
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
X = data2.drop('Class', axis=1)#independent columns
y =data2['Class'] #target column 
bestfeatures = SelectKBest(score_func=mutual_info_classif)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score'] 


# In[2]:


# Select genes based on threshold 
arr1 = dfscores.to_numpy()
arr2=featureScores['Specs']
size = len(arr1)
m1=max(arr1)
sum=0
a=[]
b=[]
for i in range (size):
        if arr1[i]>=0.3*m1:
            a.append(arr1[i])
            b.append(arr2[i])


# In[3]:


# Second gene selected method (f_classif)
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
X = data2.drop('Class', axis=1)#independent columns
y =data2['Class'] #target column 
bestfeatures = SelectKBest(score_func=f_classif)
fit = bestfeatures.fit(X,y)
dfscores1 = pd.DataFrame(fit.scores_)
dfcolumns1 = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores1 = pd.concat([dfcolumns1,dfscores1],axis=1)
featureScores1.columns = ['Specs','Score']  #naming the dataframe columns


# In[5]:


# Select number of genes based on threshold 
arr3 = dfscores1.to_numpy()
arr4=featureScores1['Specs']
size = len(arr3)
m2=max(arr3)
a1=[]
b1=[]
for i in range (size):
        if arr3[i]>=0.3*m2:
            a1.append(arr3[i])
            b1.append(arr4[i])


# In[6]:


import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X = data2.drop('Class', axis=1)#independent columns
y =data2['Class'] #target column 
bestfeatures = SelectKBest(score_func=chi2)
fit = bestfeatures.fit(X,y)
dfscores2 = pd.DataFrame(fit.scores_)
dfcolumns2 = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores2 = pd.concat([dfcolumns2,dfscores2],axis=1)
featureScores2.columns = ['Specs','Score']  #naming the dataframe columns


# In[7]:


# Select number of genes based on threshold 
arr5 = dfscores2.to_numpy()
arr6=featureScores2['Specs']
size = len(arr5)
m3=max(arr5)
a2=[]
b2=[]
for i in range (size):
        if arr5[i]>=0.3*m3:
            a2.append(arr5[i])
            b2.append(arr6[i])


# In[ ]:


# Fuzzaction by using triangular


# In[8]:


#  Mf= (wi-a)/(b-a) where wi is the score a selected gene 
# a is lowest score , b is the highest value of score
MF=[]
z=0
for i in range (len(a)):
    z=(a[i] -min(a))/(max(a)-min(a))
    MF.append(z)


# In[9]:


MF1=[]
z1=0
for j in range (len(a1)):
    z1=(a1[j] -min(a1))/(max(a1)-min(a1))
    MF1.append(z1)


# In[10]:


MF2=[]
z2=0
for k in range (len(a2)):
    z2=(a2[k] -min(a2))/(max(a2)-min(a2))
    MF2.append(z2)


# In[11]:


x=0
name=[]
score=[]
z=0
for i in range(len(MF)):
    x=0
    for j in range(len(MF1)):
        if b[i]==b1[j]:
            name.append(b[i])
            score.append(MF[i]+MF1[j])
            x=1
            break   
    if x==0:
        name.append(b[i])
        score.append(MF[i])
for k in range(len(MF1)):
    z=0
    for h in range(len(MF)):
        if b1[k]!= b[h]:
            z=z+1
    if z==len(MF):
        name.append(b1[k])
        score.append(MF1[k])
                   


# In[12]:


x1=0
final_name=[]
final_score=[]
z1=0
for i1 in range(len(score)):
    x1=0
    for j1 in range(len(a2)):
        if name[i1]==b2[j1]:
            final_name.append(name[i1])
            final_score.append(score[i1]+MF2[j1])
            x1=1
            break   
    if x1==0:
        final_name.append(name[i1])
        final_score.append(score[i1])
        
for k in range(len(MF2)):
    z1=0
    for h in range(len(score)):
        if b2[k]!= name[h]:
            z1=z1+1
    if z1==len(score): 
        final_name.append(b2[k])
        final_score.append(MF2[k])


# In[ ]:


# Defuzzication  Centre of gravity 


# In[13]:


average=0
Output=[]
for h in range(len(final_score)):
    average=final_score[h] /3
    Output.append(average)


# In[22]:


Genes=[]
Weight=[]
Threshold = max(Output)*0.5
for k in range (len(Output)):
    if Output[k]>=Threshold:
        Genes.append(final_name[k]) 
        Weight.append(Output[k]) 


# In[23]:


len(Genes)


# In[24]:


data3=data2[Genes]
data3['Class'] = y


# In[17]:


data3


# In[19]:


data3.to_csv(r'C:/Users/Mahmood/Desktop/Camparsion/LungCancer/Microarray/DataSetOfLungCancer/GSE19804/SelectedGenes.csv')


# In[25]:


import pandas as pd
data4= pd.read_csv("C:/Users/Mahmood/Desktop/Camparsion/LungCancer/Microarray/DataSetOfLungCancer/GSE19804/SelectedGenes.csv")


# In[21]:


data4


# In[26]:


from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics

X = data4.drop('Class', axis=1)#independent columns
y =data4['Class']
i=0
sum=0
# ensure all data are floating point values
X = X.astype('float32')
for i in range (20):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42) # 70% training and 30% test

        mlp = MLPClassifier(hidden_layer_sizes=(300,200,100), activation='relu', 
                            max_iter = 200, solver = 'adam')
        # Train the classifier with the traning data
        mlp.fit(X_train,y_train)
        print("Training set score: %f" % mlp.score(X_train, y_train))
        print("Test set score: %f" % mlp.score(X_test, y_test))
        sum=sum+mlp.score(X_test, y_test)
        y_pred = mlp.predict(X_test)
        print(classification_report(y_test, y_pred))
        if i==19:
            print("Overall accuracy =",sum/20)


# In[27]:


# importing necessary libraries  
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 
from sklearn import metrics
from sklearn.model_selection import train_test_split 
# X -> features, y -> label 
X = data4.drop('Class', axis=1)
y =data4['Class']
i=0
sum=0
for i in range(20):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state =42) 
  
        # training a DescisionTreeClassifier 
        from sklearn.tree import DecisionTreeClassifier 
        dtree_model = DecisionTreeClassifier(max_depth = 4).fit(X_train, y_train) 
        dtree_predictions = dtree_model.predict(X_test) 
        # creating a confusion matrix 
        cm = confusion_matrix(y_test, dtree_predictions) 
        print("Accuracy:",metrics.accuracy_score(y_test, dtree_predictions))
        sum=sum+metrics.accuracy_score(y_test, dtree_predictions)
        print(classification_report(y_test, dtree_predictions))
        if i==19:
            print("OverAll accuracy=",sum/20)


# In[28]:


# importing necessary libraries 
import pandas as pd
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
X3 = data4.drop('Class', axis=1)
y3 =data4['Class']
i=0
sum=0
for i in range(20):
        X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.3,random_state = 42) 

        # training a Naive Bayes classifier 
        from sklearn.naive_bayes import GaussianNB 
        gnb = GaussianNB().fit(X_train3, y_train3) 
        gnb_predictions = gnb.predict(X_test3) 
        # accuracy on X_test 
        accuracy = gnb.score(X_test3, y_test3) 
        sum=sum+accuracy
        # creating a confusion matrix 
        cm = confusion_matrix(y_test3, gnb_predictions) 
        from sklearn.metrics import classification_report
        print(classification_report(y_test3, gnb_predictions))
        if i==19:
            print("Overall accuracy=", sum/20)


# In[29]:


import pandas as pd
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
X3 = data4.drop('Class', axis=1)
y3 =data4['Class'] 

# dividing X, y into train and test data 
i=0
sum=0
for i in range(20):
        X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.3,random_state = 42) 
        fanr = RandomForestClassifier().fit(X_train3, y_train3) 
        fanr_predictions = fanr.predict(X_test3) 
        # accuracy on X_test 
        accuracy = fanr.score(X_test3, y_test3) 
        sum=sum+ accuracy
        # creating a confusion matrix 
        cm = confusion_matrix(y_test3, fanr_predictions) 
        print(classification_report(y_test3, fanr_predictions))
        if i==19:
            print("overall Accuracy=",sum/20)
    


# In[30]:


# importing necessary libraries 

from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
import pandas as pd
X2 = data4.drop('Class', axis=1)
y2 =data4['Class']
i=0
sum=0
for i in range(20):
    # dividing X, y into train and test data 
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2,test_size=0.3, random_state = 42) 

    # training a KNN classifier 
    from sklearn.neighbors import KNeighborsClassifier 
    knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train2, y_train2) 

    # accuracy on X_test 
    accuracy = knn.score(X_test2, y_test2) 
    sum=sum+accuracy 

    # creating a confusion matrix 
    knn_predictions = knn.predict(X_test2)  
    cm = confusion_matrix(y_test2, knn_predictions) 
    from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
    print("Accuracy:",metrics.accuracy_score(y_test2, knn_predictions))
    from sklearn.metrics import classification_report
    print(classification_report(y_test2 ,knn_predictions))
    if i==19:
        print("Overall accuracy",sum/20)


# In[31]:


# importing necessary libraries 
import pandas as pd
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split  
X1 = data4.drop('Class', axis=1)
y1 =data4['Class']
i=0
sum=0
for i in range (20):
    # dividing X, y into train and test data 
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1,test_size=0.3, random_state = 42) 

    # training a linear SVM classifier 
    from sklearn.svm import SVC 
    svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train1, y_train1) 
    svm_predictions = svm_model_linear.predict(X_test1) 
    accuracy = svm_model_linear.score(X_test1, y_test1)  
    
    # creating a confusion matrix 
    cm = confusion_matrix(y_test1, svm_predictions) 
    from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
    print("Accuracy:",metrics.accuracy_score(y_test1, svm_predictions))
    sum=sum+metrics.accuracy_score(y_test1, svm_predictions)
    from sklearn.metrics import classification_report
    print(classification_report(y_test1, svm_predictions))
    if i==19:
        print("Overall accuracy=",sum/20)
    


# In[ ]:




