#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

import textwrap
from transformers import pipeline, set_seed
from pprint import pprint

import warnings
warnings.filterwarnings('ignore')


# In[4]:


text = ('The Book of Genesis is the first book of the Hebrew Bible and the Christian Old Testament. It serves as the opening chapter of both religious texts and is foundational to the Abrahamic faiths, including Judaism, Christianity, and Islam. Genesis is a compilation of ancient narratives that explore the creation of the world, the origins of humanity, and the early history of the Israelite people. Key themes in Genesis include the creation of the universe in six days, the fall of humanity through Adam and Eves disobedience, the stories of prominent figures such as Noah and the flood, and the patriarchs Abraham, Isaac, and Jacob. The book also recounts the migration of the Israelites to Egypt and introduces Joseph, whose rise to power in Egypt sets the stage for the enslavement of the Israelites.Genesis provides a theological and historical foundation for the subsequent books of the Bible, establishing fundamental concepts like the covenant between God and humanity, the chosen people, and the promise of a land. The narrative style combines mythic elements with historical accounts, offering a rich and complex portrayal of the origins of the world and the people of Israel.')


# In[5]:


print(text)


# ### Text Generation :

# Text generation using transformers relies on sophisticated neural network architectures, like the Transformer model, which utilizes self-attention mechanisms to understand and produce coherent text by capturing contextual dependencies within input sequences. The popularity of transformers stems from their ability to parallelize computations, facilitating quicker training on extensive datasets. Their strength lies in effectively capturing long-range dependencies, enabling the generation of contextually rich and relevant output. Transformer-based text generation finds applications in diverse fields, encompassing natural language understanding, language translation, and creative writing. 

# In[6]:


gen = pipeline('text-generation')


# In[7]:


gen(text)


# In[14]:


gen(text[1])


# In[16]:


pprint(gen(text, num_return_sequences = 3, max_length = 20))


# In[17]:


def wrap(x):
    return textwrap.fill(x, 
                         replace_whitespace = False, 
                         fix_sentence_endings = True)


# In[19]:


out = gen(text, max_length = 30)
generated_text = out[0]['generated_text']
print(wrap(generated_text))


# In[21]:


prompt = 'Neural networks with attention have been used with great success' + \
'in natural language processing.'

out = gen(prompt, max_length = 150)
print(wrap(out[0]['generated_text']))


# In[ ]:




