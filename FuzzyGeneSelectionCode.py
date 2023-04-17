#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Loading Dataset
import pandas as pd
data2 = pd.read_csv("C:/Users/Mahmood/Desktop/Camparsion/LungCancer/Microarray/OrginalData/GSE19804/Preprossing/DataStructured3.csv")


# In[2]:


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


# In[ ]:


# Select genes based on Step Function (SF)
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


# In[ ]:


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


# In[ ]:


# Select number of genes based on Step Function 
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


# In[ ]:


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


# In[ ]:


# Select number of genes based on Step Function 
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


# In[ ]:


#  Mf= (wi-a)/(b-a) where wi is the score a selected gene 
# a is lowest score , b is the highest value of score
MF=[]
z=0
for i in range (len(a)):
    z=(a[i] -min(a))/(max(a)-min(a))
    MF.append(z)


# In[ ]:


MF1=[]
z1=0
for j in range (len(a1)):
    z1=(a1[j] -min(a1))/(max(a1)-min(a1))
    MF1.append(z1)


# In[ ]:


MF2=[]
z2=0
for k in range (len(a2)):
    z2=(a2[k] -min(a2))/(max(a2)-min(a2))
    MF2.append(z2)


# In[ ]:


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
                   


# In[ ]:


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


# In[ ]:


average=0
Output=[]
for h in range(len(final_score)):
    average=final_score[h] /3
    Output.append(average)


# In[ ]:


Genes=[]
Weight=[]
SF = max(Output)*0.5
for k in range (len(Output)):
    if Output[k]>=SF:
        Genes.append(final_name[k]) 
        Weight.append(Output[k]) 


# In[ ]:


len(Genes)


# In[ ]:


data3=data2[Genes]
data3['Class'] = y


# In[ ]:


data3.to_csv(r'C:/Users/Mahmood/Desktop/Camparsion/LungCancer/Microarray/DataSetOfLungCancer/GSE19804/SelectedGenes.csv')

