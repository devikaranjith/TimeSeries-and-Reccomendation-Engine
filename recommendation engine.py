#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[13]:


df=pd.read_csv(r"C:\Users\santr\Downloads\article_info.csv")
df


# In[14]:


df=df.drop(['website','content'],axis=1) 


# In[15]:


df.duplicated().sum()


# In[16]:


df


# In[17]:


vectorizer = TfidfVectorizer()

# fit the vectorizer to the data
tfidf_matrix = vectorizer.fit_transform(df['title'])


# In[18]:


cosine_sim = cosine_similarity(tfidf_matrix)


# In[19]:


def recommend_articles(title, cosine_sim=cosine_sim):
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    return df['title'].iloc[ [i[0] for i in sim_scores[1:11]]]


# In[20]:


title="Tomber pour les composants Web"
recommend_articles(title)


# In[ ]:




