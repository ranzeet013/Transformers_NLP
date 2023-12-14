#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from transformers import pipeline 

import warnings
warnings.filterwarnings('ignore')


# ### Zero-shot Classification :

# `zero-shot-classification` is an NLP pipeline that performs text classification without task-specific training. It requires candidate labels and predicts if the input text is related to any of them. The model is pre-trained and versatile, making it suitable for various classification tasks. It offers a quick solution when specific labeled data is unavailable. It's part of the Hugging Face Transformers library.

# In[3]:


classifier = pipeline('zero-shot-classification')


# In[8]:


text = 'The anime was awesome'
candidate_labels = ['Negative', 'Positive']

result = classifier(text, candidate_labels=candidate_labels)
print(result)


# In[4]:


text = 'Eren Yeager is a boy who lives in the town of Shiganshina, located on the outermost of three circular walls which' + \
        'protect their inhabitants from Titans. In the year 845, the first wall (Wall Maria) is breached by two new types of ' + \
        'Titans, the Colossal Titan and the Armored Titan. During the incident, Eren\'s mother is eaten by a Smiling Titan while' +\
        ' Eren escapes. He swears revenge on all Titans and enlists in the military along with his childhood friends Mikasa Ackerman and Armin Arlert.'


# In[5]:


print(text)


# In[10]:


classifier(text, candidate_labels = ['Movie', 'Anime'])


# In[ ]:




