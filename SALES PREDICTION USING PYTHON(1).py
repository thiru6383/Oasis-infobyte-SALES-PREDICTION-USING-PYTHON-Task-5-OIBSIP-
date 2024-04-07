#!/usr/bin/env python
# coding: utf-8

# # SALES PREDICTION USING PYTHON:
# 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model


# In[2]:


data_set=pd.read_csv(r"C:\Users\HP\OneDrive\Documents\mohankumar\Advertising1.csv")


# In[3]:


data_set.head()


# In[4]:


data_set.tail()


# In[5]:


data_set.shape


# In[6]:


data_set.isnull().sum() #checking missing values.


# In[7]:


data_set.describe()


# In[8]:


data_set.info()


# In[9]:


data_set.columns


# In[ ]:





# In[ ]:





# # Data visualization

# In[22]:


sns.barplot(data_set)
plt.show()


# In[11]:


sns.pairplot(data_set)
plt.show()


# In[12]:


data_set.hist()
plt.show()


# In[13]:


sns.heatmap(data_set.corr(),cmap='Greens')


# In[ ]:





# In[ ]:





# # Training the model

# In[15]:


inputs=data_set.drop(['Sales'],axis='columns')
inputs


# In[16]:


Target=data_set.Sales
Target


# In[17]:


model=linear_model.LinearRegression()
inputs=inputs.values 


# In[18]:


model.fit(inputs,Target)


# In[ ]:





# In[ ]:





# # Final Result of Sales prediction using python:

# In[27]:


prediction=model.predict( [[3.4,5.2,7.5 ]])              #input: ( T.V, Radio, Newspapper.)
print("our sales prediction is",prediction)


# In[ ]:





# In[20]:


# Thanking you...


#                                                                                               
