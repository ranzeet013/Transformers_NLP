#!/usr/bin/env python
# coding: utf-8

# In[1]:


article = 'Ukraine Aid Falters in U.S. Senate as Republicans Insist on Border Restrictions Legislation to send military aid to Ukraine and Israel was on the brink of collapse after a briefing devolved into a screaming match before a critical vote.'


# In[2]:


import numpy as np
import pandas as pd

import textwrap
from transformers import pipeline


# ### Text Summarization :

# Text summarization is a natural language processing technique aimed at condensing large volumes of text while retaining its key information. There are two primary types: extractive summarization, which involves selecting and presenting existing sentences, and abstractive summarization, which generates new, concise sentences to convey the main ideas. Extractive methods often use algorithms to identify and rank important sentences based on content relevance, while abstractive methods employ advanced language models to create concise summaries that may not directly mirror the original wording. 

# In[3]:


summarizer = pipeline('summarization')


# In[4]:


def wrap(x):
    return textwrap.fill(x, 
                         replace_whitespace = False,
                         fix_sentence_endings = True)

print(wrap(article))


# In[5]:


summarizer(wrap(article))


# In[ ]:




