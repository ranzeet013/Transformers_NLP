#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as npp

from transformers import AutoTokenizer

import warnings
warnings.filterwarnings('ignore')


# In[3]:


checkpoint = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# ### Tokenization :

# Tokenization is the process of breaking down a sequence of text into smaller units, called tokens. Tokens are the building blocks of natural language processing (NLP) tasks, and they can represent words, subwords, characters, or other meaningful units depending on the context and the chosen tokenizer. Tokenization is a crucial step in processing and analyzing text data in various NLP applications.

# In[4]:


tokenizer('hello world')


# In[5]:


tokens = tokenizer.tokenize('hello world')


# In[6]:


tokens


# In[7]:


ids = tokenizer.convert_tokens_to_ids(tokens)
ids


# In[8]:


tokenizer.decode(ids)


# In[9]:


ids = tokenizer.encode('hello world')
ids


# In[11]:


tokenizer.convert_ids_to_tokens(ids)


# In[12]:


model_inputs = tokenizer('hello world')
model_inputs


# In[16]:


data = [
    'I like Animes.', 
    'Do you like Animes too?',
]
tokenizer(data)


# ### Models in NLP :

# NLP models, ranging from rule-based to advanced deep learning architectures like BERT and GPT, aim to understand and process human language. Statistical models and machine learning classifiers, such as SVM and Naive Bayes, were early contributors. Recent breakthroughs like Transformers have dominated NLP tasks, utilizing attention mechanisms for improved performance. Pre-trained models like BERT and GPT offer transfer learning capabilities, while multimodal models integrate language understanding with vision or audio processing for diverse applications. 

# AutoModelForSequenceClassification is a class that automatically loads a pre-trained model for sequence classification tasks. Sequence classification involves assigning a label to an input sequence, such as classifying a document into different categories or determining the sentiment of a text.

# In[17]:


from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)


# In[18]:


model_inputs = tokenizer('hello world', return_tensors = 'pt')
model_inputs


# In[20]:


outputs = model(**model_inputs)
outputs


# In[21]:


model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 3)


# In[22]:


outputs = model(**model_inputs)
outputs


# In[24]:


outputs.logits


# In[26]:


data = [
    'I like Animes.', 
    'Do you like Animes too?',
]
model_inputs = tokenizer(data,padding = True, truncation = True, return_tensors = 'pt')
model_inputs


# In[27]:


model_inputs['input_ids']


# In[28]:


model_inputs['attention_mask']


# In[29]:


outputs = model(**model_inputs)
outputs


# In[ ]:




