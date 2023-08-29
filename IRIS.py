#!/usr/bin/env python
# coding: utf-8

# # Lets Grow More(LGMVIP)
# ## Task 1: Iris flower Classification ML Project 
# ### Author: Omkar Sanjay Chavan

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# # Loading Data

# In[2]:


data=pd.read_csv("D:/Omkar/Iris.csv");data


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data[('Species')].unique()


# In[9]:


data.info()


# In[6]:


iris=pd.DataFrame(data);iris


# In[7]:


fig = iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=fig)
iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length VS Width")
fig=plt.gcf()
plt.legend(bbox_to_anchor=(1, 1), loc=2)
fig.set_size_inches(5,3)
plt.show()


# In[8]:


fig=iris[iris.Species=='Iris-setosa'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='Black',label='Setosa')
iris[iris.Species=='Iris-versicolor'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='blue',label='versicolor',ax=fig)
iris[iris.Species=='Iris-virginica'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',color='orange',label='virginica',ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title("Petal Length VS Width")
fig=plt.gcf()
plt.legend(bbox_to_anchor=(1, 1), loc=2)
fig.set_size_inches(6,3)
plt.show()


# From the above diagram we can conclude that setosa has the lowest petal width and petal length and there is positive correlation between them for setosa. viginica has highest petal width and petal length and there is positive correlation between them for virginica.

# In[9]:


plt.figure(figsize = (5, 4))
x = data["SepalLengthCm"]
  
plt.hist(x, bins = 20, color = "maroon")
plt.title("Sepal Length in cm")
plt.xlabel("Sepal_Length_cm")
plt.ylabel("Count")


# In[11]:


plt.figure(figsize = (5, 4))
x = data["SepalWidthCm"]
  
plt.hist(x, bins = 20, color = "maroon")
plt.title("Sepal Width in cm")
plt.xlabel("Sepal_Width_cm")
plt.ylabel("Count")


# In[12]:


fig, axes = plt.subplots(2, 2, figsize=(10,10))
 
axes[0,0].set_title("Sepal Length")
axes[0,0].hist(data['SepalLengthCm'], bins=5)
 
axes[0,1].set_title("Sepal Width")
axes[0,1].hist(data['SepalWidthCm'], bins=6);
 
axes[1,0].set_title("Petal Length")
axes[1,0].hist(data['PetalLengthCm'], bins=6);
 
axes[1,1].set_title("Petal Width")
axes[1,1].hist(data['PetalWidthCm'], bins=6);


# In[13]:


corr=iris.corr()
sns.heatmap(corr,annot=True)


# From the above heatmap diagram we can see that there is high positive correlation between Petal width and petal length. 

# In[15]:


import numpy as np
iris.groupby('Species').agg(['mean','median'])
iris.groupby('Species').agg([np.mean, np.median])


# In[16]:


X=iris.drop(['Species'],axis=1);iris


# In[17]:


Y=iris['Species'];Y


# In[18]:


X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)


# In[19]:


model=LogisticRegression(solver='lbfgs',max_iter=3000)


# In[20]:


model.fit(X_train,y_train)


# In[21]:


print("Accuracy score is ",model.score(X_test,y_test))


# In[22]:


from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix,accuracy_score,classification_report


# In[23]:


y_pred=model.predict(X_test)
y_pred


# In[24]:


print("The f1 score is ",f1_score(y_test,y_pred,average='micro'))


# In[25]:


print("the confusion matrix is\n", confusion_matrix(y_test,y_pred))


# In[27]:


model.predict([[4.0,2.0,3.2,1.1,1.1]])

