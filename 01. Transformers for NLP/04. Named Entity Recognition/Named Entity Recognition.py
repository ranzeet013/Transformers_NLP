#!/usr/bin/env python
# coding: utf-8

# In[1]:


sentences = [
    "Apple is a technology company.",
    "John and Mary visited New York City.",
    "Python is a popular programming language.",
    "The Eiffel Tower is located in Paris.",
    "Elon Musk founded SpaceX.",
]


# In[2]:


from transformers import pipeline

import pickle


# ### NER, or Named Entity Recognition :

# NER, or Named Entity Recognition, is a natural language processing (NLP) technique that involves identifying and classifying entities, such as names of people, organizations, locations, dates, and other specific terms, within a text. The goal of NER is to extract meaningful information and provide a structured representation of the text by labeling entities with predefined categories. This technology is crucial for various applications, including information retrieval, question answering, and sentiment analysis, as it enables machines to understand and process text at a more granular level, improving overall comprehension and usability of language-based systems.

# In[4]:


ner = pipeline('ner', aggregation_strategy = 'simple')


# In[5]:


ner(sentences)


# In[ ]:




