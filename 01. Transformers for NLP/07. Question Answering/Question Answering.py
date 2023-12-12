#!/usr/bin/env python
# coding: utf-8

# In[2]:


from transformers import pipeline


# In[3]:


qa = pipeline('question-answering')


# In[4]:


context = 'Today I went to walmart to buy PS5.'

question = 'What did i buy?'


# In[5]:


qa(context = context, question = question)


# In[7]:


context = 'J. Robert Oppenheimer (1904–1967) was an American theoretical physicist and scientific director of the Manhattan Project, the World War II initiative that developed the first nuclear weapons. Born in New York City, Oppenheimer exhibited exceptional academic prowess, earning a reputation as a brilliant scholar. He made significant contributions to theoretical physics, particularly in quantum mechanics and quantum electrodynamics. However, Oppenheimer is best known for his role in leading the Los Alamos Laboratory, where he oversaw the development of the atomic bomb during the early 1940s.'


# In[8]:


print(context)


# In[9]:


question = 'Who was J. Robert Oppenheimer?'


# In[10]:


qa(context = context, question = question)


# In[ ]:




