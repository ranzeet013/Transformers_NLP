#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import textwrap 

from transformers import pipeline 
from pprint import pprint 

import warnings
warnings.filterwarnings('ignore')


# In[2]:


article = ('Ukraine Aid Falters in Senate as Republicans Insist on Border Restrictions')


# ### Masked Language Modeling : 

# 
# Masked Language Modeling (MLM) is a type of language modeling task used in natural language processing (NLP) and machine learning. In MLM, certain words in a given text are randomly masked or replaced with a special token. The model is then trained to predict the original words based on the context provided by the surrounding words. This helps the model learn the relationships between words and improve its understanding of language semantics. MLM is a key component in pre-training language models, where a model is first trained on a large corpus of text before being fine-tuned for specific downstream tasks. It has been particularly successful in the development of state-of-the-art models for various NLP applications.

# In[3]:


mask_language_modeling  = pipeline('fill-mask')


# In[5]:


mask_language_modeling('Ukraine Aid Falters in Senate as <mask> Insist on Border Restrictions')


# In[9]:


text = 'Ukraine Aid Falters in Senate as <mask> Insist on Border Restrictions ' + \
       'Legislation to send military aid to Ukraine and Israel was on the brink of collapse ' + \
       'after a briefing devolved into a screaming match one day before a critical test vote in the Senate.'


# In[10]:


mask_language_modeling(text)


# In[12]:


text = 'Ukraine Aid Falters in Senate as <mask> Insist on Border Restrictions ' + \
       'Legislation to send military aid to Ukraine and Israel was on the brink of collapse ' + \
       'after a briefing devolved into a <mask> match one day before a critical test vote in the Senate.'

pprint(mask_language_modeling(text))

