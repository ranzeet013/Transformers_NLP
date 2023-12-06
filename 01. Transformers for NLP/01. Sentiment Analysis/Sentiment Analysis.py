#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries :

# I have downloaded all the libraries required for this sentiment analysis project.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

import torch
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score

import warnings
warnings.filterwarnings('ignore')


# In[2]:


classifier = pipeline('sentiment-analysis')
classifier


# In[3]:


classifier(['The movie glorifies violence, physical and sexual abuse, denigration of women and basically banks on shock value to provoke viewers rather than pull in people with a coherent story.', 
            'As a fan of gore and dark noir went with loads of expectations- first half was off to a near flawless flight but then the tailspin which began post-interval took down the whole movie.'])


# ### Importing the Dataset :

# In[4]:


dataframe = pd.read_csv('Tweets.csv')


# In[5]:


dataframe.head()


# In[6]:


dataframe.columns


#  

# In[7]:


dataframe = dataframe[['text', 'airline_sentiment']].copy()


# In[8]:


dataframe.head()


# Calculates and prints the value counts of the 'airline_sentiment' column in the DataFrame.
# 
# Creating a bar plot to visualize the distribution of sentiment in the 'airline_sentiment' column with a specified title and figsize.

# In[9]:


value_counts = dataframe['airline_sentiment'].value_counts()
print('Value Count For: ', value_counts)

value_counts = dataframe['airline_sentiment'].value_counts().plot(kind = 'bar', 
                                                                  title = 'Distribution of Sentiment', 
                                                                  figsize = (7, 3))


# Removing the rows where the 'airline_sentiment' column is equal to 'neutral' and creates a copy of the resulting DataFrame.Since, our model does not work with neutral data.

# In[10]:


dataframe = dataframe[dataframe.airline_sentiment != 'neutral'].copy()
dataframe.head()


# In[11]:


fig, ax = plt.subplots(figsize = (7, 3))

dataframe['airline_sentiment'].hist(ax = ax)

ax.set_xlabel('Sentiment')
ax.set_ylabel('Count')
ax.set_title('Distribution of Sentiment')

plt.show()


# mapping the positive value with 1 and negativewith 0.

# In[12]:


target_map = {'positive':'1', 'negative':'0'}
dataframe['target'] = dataframe['airline_sentiment'].map(target_map)
dataframe.head()


# Predicting the sentiment of the text

# In[13]:


text = dataframe['text'].tolist()
predictions = classifier(text)


# In[14]:


predictions


# In[15]:


probs = [d['score'] if d['label'].startswith('P') else 1 - d['score'] \
         for d in predictions]
probs


# In[16]:


preds = [1 if d['label'].startswith('P') else 0 for d in predictions]
preds = np.array(preds)
preds


# Calculating the accuracy of the model by comparing its predicted labels with true labels after converting it to integer.

# In[19]:


true_labels = dataframe['target'].astype(int).values

accuracy = np.sum(preds == true_labels) / len(true_labels)

print(f'Accuracy: {accuracy * 100:.2f}%')


# In[20]:


confusion_matrix = confusion_matrix(true_labels, preds)

print("Confusion Matrix:")
print(confusion_matrix)


# In[29]:


plt.figure(figsize = (5, 3))
sns.heatmap(confusion_matrix, 
            annot = True, 
            cmap = 'RdPu')


# In[30]:


f1_score = f1_score(true_labels, preds)


# In[31]:


f1_score


# In[33]:


roc_auc_score(true_labels, preds)


# In[34]:


print(classification_report(true_labels, preds))


# In[ ]:




