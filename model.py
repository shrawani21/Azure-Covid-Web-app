#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


# In[2]:


data = pd.read_csv("corona_tested2.csv")


# In[3]:


data1 = data.drop("test_date",axis="columns")


# In[4]:


data2 = data1.dropna(axis="rows")


# In[5]:


data2["age_60_and_above"].replace({"Yes":1,"No":0},inplace = True)
data2["corona_result"].replace({"positive":1,"negative":0, "other":1 },inplace = True)
data3 = pd.get_dummies(data = data2,columns = ['test_indication','gender'])


# In[6]:


data4 = data3.drop("gender_female",axis="columns")


# In[7]:


data4 = data4.drop("gender_male",axis="columns")


# In[8]:


c0,c1 = data4["corona_result"].value_counts()

dfc0 = data4[data4["corona_result"]==0]
dfc1 = data4[data4["corona_result"]==1]


# In[9]:


dfc0u = dfc0.sample(c1)

df1 = pd.concat([dfc0u,dfc1],axis=0)


# In[10]:


dftrainy = df1["corona_result"]
dftrainx = df1.drop("corona_result",axis="columns")


# In[11]:


from xgboost import XGBClassifier

X1model = XGBClassifier()


# In[12]:


X1model.fit(dftrainx, dftrainy)


# In[13]:


pickle.dump(X1model,open("model.pkl","wb"))


# In[14]:


model=pickle.load(open("model.pkl","rb"))


# In[15]:


test = dftrainx[:1]


# In[16]:


test.columns


# In[17]:


datak = {'cough':[1],'fever':[0],'sore_throat':[0],'shortness_of_breath':[1],'head_ache':[1],'age_60_and_above':[0],'test_indication_Abroad':[0],'test_indication_Contact':[0],'test_indication_Other':[1]}
  
# Create DataFrame
dfk = pd.DataFrame(datak)


# In[18]:


print(model.predict(dfk))


# In[ ]:



