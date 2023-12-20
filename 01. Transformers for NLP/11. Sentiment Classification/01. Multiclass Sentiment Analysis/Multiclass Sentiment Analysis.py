#!/usr/bin/env python
# coding: utf-8

# Importing necessary libraries and setting up the environment for a machine learning project. Using Pandas, NumPy, Seaborn, and Matplotlib for data manipulation and visualization. Employing PyTorch, Scikit-Learn, Hugging Face Transformers, and datasets library for machine learning tasks. AutoTokenizer and AutoModelForSequenceClassification from Transformers facilitate tokenization and model loading. The Trainer and TrainingArguments are employed for model training. The torchinfo library is used for summarizing model information. Warnings are suppressed using the warnings library. Overall, the environment is configured for efficient data processing, model training, and evaluation.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

get_ipython().run_line_magic('matplotlib', 'inline')

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, confusion_matrix
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from torchinfo import summary

import warnings
warnings.filterwarnings('ignore')


# In[2]:


dataframe = pd.read_csv('Tweets.csv')


# In[3]:


dataframe.head()


# In[4]:


dataframe.columns


# In[5]:


dataframe = dataframe[['airline_sentiment', 'text']].copy()


# In[6]:


dataframe.head()


# Sentiment labels in 'airline_sentiment' are counted and printed, showing their distribution. A bar plot visually represents this distribution with sentiment labels on the x-axis and their frequencies on the y-axis.

# In[7]:


value_counts = dataframe['airline_sentiment'].value_counts()
print('Value Count For: ', value_counts)

value_counts = dataframe['airline_sentiment'].value_counts().plot(kind = 'bar', 
                                                                  title = 'Distribution of Sentiment', 
                                                                  figsize = (7, 3))


# Sentiment labels in the 'airline_sentiment' column are converted to numerical values using the map function, creating a new 'target' column. 'Positive' is mapped to '1', 'negative' to '0', and 'neutral' to '2'. 

# In[8]:


target_map = {'positive':'1', 'negative':'0', 'neutral':'2'}
dataframe['target'] = dataframe['airline_sentiment'].map(target_map)
dataframe.head()


# Creating histogram using Matplotlib to visualize the distribution of sentiment labels in the 'airline_sentiment' column of the DataFrame. The figure and axes are set up with a specified size. The `hist` function is then applied to the 'airline_sentiment' column, and the x-axis label, y-axis label, and title are set for better interpretation. 

# In[9]:


fig, ax = plt.subplots(figsize = (7, 3))

dataframe['airline_sentiment'].hist(ax = ax)

ax.set_xlabel('Sentiment')
ax.set_ylabel('Count')
ax.set_title('Distribution of Sentiment')

plt.show()


# In[10]:


dataset = dataframe[['text', 'target']]
dataset.columns = ['sentence', 'label']
dataset.to_csv('data.csv', index = None)


# In[11]:


data = load_dataset('csv', data_files = 'data.csv')
data


# In[12]:


split = data['train'].train_test_split(test_size = 0.3, seed = 42)
split


# Loading a pre-trained DistilBERT model is accomplished by specifying the model checkpoint as 'distilbert-base-cased'. The Hugging Face `AutoTokenizer` is then used to create a tokenizer for the chosen checkpoint. This tokenizer is crucial for converting raw text data into tokenized sequences compatible with the DistilBERT model.

# In[13]:


checkpoint = 'distilbert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# Tokenizing the dataset involves defining a `tokenize_function` that utilizes the Hugging Face tokenizer with truncation. The function processes batches of sentences from the dataset. The actual tokenization is applied using `split.map`, ensuring efficient batch-wise processing. This step is essential for converting raw text into tokenized representations suitable for training a machine learning model.

# In[14]:


def tokenize_function(batch):
    return tokenizer(batch['sentence'], truncation = True)

tokenized_dataset = split.map(tokenize_function, batched = True)


# In[15]:


model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 3)


# In[16]:


summary(model)


# Training parameters for the Hugging Face Trainer involves setting key values. The output_dir designates the directory for storing model checkpoints and outputs. The evaluation_strategy and save_strategy are both set to 'epoch,' indicating evaluation and checkpoint saving at each epoch's end. num_train_epochs is set to 1, determining the total training epochs. per_device_train_batch_size and per_device_eval_batch_size are set to 16 and 64, respectively, defining batch sizes for training and evaluation on each device.

# In[17]:


training_args = TrainingArguments(
    output_dir = 'training_dir',
    evaluation_strategy = 'epoch', 
    save_strategy = 'epoch', 
    num_train_epochs = 1, 
    per_device_train_batch_size = 16, 
    per_device_eval_batch_size = 64
    
)


# Defining a `compute_metrics` function for use with the Hugging Face `Trainer`, this function takes the output of the model predictions and true labels as input and computes accuracy and F1 score. The `Trainer` is then initialized with the model, training arguments, tokenized training and evaluation datasets, tokenizer, and the defined `compute_metrics` function. The training process is executed using the `trainer.train()` method.

# In[18]:


def compute_metrics(p):
    logits, labels = p
    predictions = np.argmax(logits, axis=-1)
    acc = np.mean(predictions == labels)
    f1 = f1_score(labels, predictions, average='macro')
    
    return {
        'accuracy': acc,
        'f1_score': f1,
    }

trainer = Trainer(
    model, 
    training_args, 
    train_dataset=tokenized_dataset['train'], 
    eval_dataset=tokenized_dataset['test'], 
    tokenizer=tokenizer, 
    compute_metrics=compute_metrics 
)

trainer.train()


# In[ ]:




