#!/usr/bin/env python
# coding: utf-8

# ### Fine Tuning Sentiment Analysis 

# ### Importing Libraries :

# Importing required libraries and modules, including NumPy, Hugging Face Transformers, Datasets, pprint, TorchInfo, and the Trainer module from Transformers for sequence classification.

# In[1]:


import numpy as np

from transformers import AutoTokenizer
from datasets import load_dataset
from pprint import pprint
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from torchinfo import summary
from transformers import Trainer
from datasets import load_metric


# In[2]:


dataframe = load_dataset('rotten_tomatoes')


# In[3]:


dataframe


# In[4]:


dataframe['train']


# In[5]:


dir(dataframe['train'])


# In[6]:


dataframe.data


# In[7]:


dataframe['train'][0]


# In[8]:


dataframe['train'].features


# Loading a tokenizer for the 'distilbert-base-uncased' model using the Hugging Face Transformers library. The tokenizer is created with `AutoTokenizer.from_pretrained(checkpoint)`.

# In[9]:


checkpoint = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# Tokenizing the text content of the first three rows in the 'text' column of the 'train' subset of the dataframe using the tokenizer and printing the tokenized result.

# In[10]:


sentence_tokenized = tokenizer(dataframe['train'][0:3]['text'])

pprint(sentence_tokenized)


# Defining a `tokenize_function` that tokenizes text in batches from a dataframe using the provided tokenizer with truncation enabled. Applying this function to a dataframe (`dataframe`) using the `map` method with batch processing (`batched=True`), and storing the tokenized result in `tokenized_dataframe`.

# In[11]:


def tokenize_function(batch):
    return tokenizer(batch['text'], truncation=True)

tokenized_dataframe = dataframe.map(tokenize_function, batched=True)


# Creating a `TrainingArguments` object named `training_args` with the output directory set to 'trainer_log', evaluation strategy set to 'epoch', save strategy set to 'epoch', and training for one epoch (`num_train_epochs = 1`).

# In[12]:


training_args = TrainingArguments(
    'trainer_log', 
    evaluation_strategy = 'epoch', 
    save_strategy = 'epoch', 
    num_train_epochs = 1
)


# In[13]:


model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels =  2)


# In[14]:


summary(model)


# Iterating through the named parameters of a PyTorch model, `model`, and storing their detached numpy values in the list `params_before`.

# In[15]:


params_before = []
for name, p in model.named_parameters():
    params_before.append(p.detach().numpy())


# Defining a `metric_function` to calculate accuracy between predictions and references using `accuracy_score`. Creating a `compute_metrics` function that computes accuracy based on logits and labels, and configuring a `Trainer` instance with training and evaluation datasets, tokenizer, and metric computation. Initiating training using `trainer.train()`.

# In[17]:


from sklearn.metrics import accuracy_score

def metric_function(predictions, references):
    return accuracy_score(references, predictions)

def compute_metrics(logits_and_labels):
    logits, labels = logits_and_labels 
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    
    return {"accuracy": accuracy}


trainer = Trainer(
    model, 
    training_args, 
    train_dataset=tokenized_dataframe['train'], 
    eval_dataset=tokenized_dataframe['validation'], 
    tokenizer=tokenizer, 
    compute_metrics=compute_metrics
)

trainer.train()


# In[18]:


trainer.save_model('model.h5')


# In[20]:


from transformers import pipeline

classifier = pipeline('text-classification', model = 'model.h5')


# In[21]:


classifier('That movie was fucking awesome.')


# In[ ]:




