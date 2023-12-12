#!/usr/bin/env python
# coding: utf-8

# ### Neural Machine Translation (NMT) :

# Neural Machine Translation (NMT) is a deep learning approach for automatically translating text from one language to another. It utilizes neural networks, particularly recurrent or transformer architectures, to learn complex mappings between source and target languages. NMT considers the entire input sentence at once, capturing contextual dependencies and producing more fluent translations compared to traditional methods. Training involves large parallel corpora, enabling the model to generalize across diverse language pairs. NMT has become the dominant paradigm in machine translation, offering improved translation quality and natural language understanding.

# In[1]:


import pandas as pd
import numpy as np

from transformers import pipeline 
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import RegexpTokenizer

# Initialize a RegexpTokenizer for word tokenization
tokenizer = RegexpTokenizer(r'\w+')

# Dictionary to store English to Spanish translations
eng2spa = {}

# Read the English to Spanish translation data from a file
with open(r'C:/Users/DELL/Desktop/python project/nlp/New folder (6)/spa-eng/spa.txt', 'r', encoding='utf-8') as file:
    for line in file:
        line = line.rstrip()
        eng, spa = line.split("\t")
        eng2spa[eng] = spa

# Example usage of sentence_bleu
sentence_bleu([['hi']], ['hi'])

# Smoothing the BLEU score using NLTK's SmoothingFunction
smoother = SmoothingFunction()
sentence_bleu([['hi']], ['hi'], smoothing_function=smoother.method4)

# Tokenize the Spanish translations
eng2spa_tokens = {}
for eng, spa_list in eng2spa.items():
    spa_list_tokens = []
    for text in spa_list:
        tokens = tokenizer.tokenize(text.lower())
        spa_list_tokens.append(tokens)
    eng2spa_tokens[eng] = spa_list_tokens
    
# Initialize a translation pipeline using the Helsinki-NLP model for English to Swedish translation
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-se")

translator('I am ranzeet.') # translates into spanish text 


# In[ ]:




