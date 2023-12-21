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

dataframe.head()


# In[3]:


dataframe = dataframe[['airline_sentiment', 'text']].copy()

dataframe.head()


# In[4]:


value_counts = dataframe['airline_sentiment'].value_counts()
print('Value Count For: ', value_counts)

value_counts = dataframe['airline_sentiment'].value_counts().plot(kind = 'bar', 
                                                                  title = 'Distribution of Sentiment', 
                                                                  figsize = (7, 3))


# Filtering a DataFrame to exclude entries with 'neutral' sentiment, this Python script then computes and prints the value counts for each sentiment category. The resulting counts are visualized using a bar plot to illustrate the distribution of sentiments in the dataset.

# In[5]:


dataframe = dataframe[dataframe['airline_sentiment'] != 'neutral']

value_counts = dataframe['airline_sentiment'].value_counts()
print('Value Count For: ', value_counts)

value_counts.plot(kind='bar', title='Distribution of Sentiment', figsize=(7, 3))
plt.show()


# In[6]:


target_map = {'positive':'1', 'negative':'0'}
dataframe['target'] = dataframe['airline_sentiment'].map(target_map)
dataframe.head()


# In[7]:


dataset = dataframe[['text', 'target']]
dataset.columns = ['sentence', 'label']
dataset.to_csv('data.csv', index = None)


# In[8]:


data = load_dataset('csv', data_files = 'data.csv')
data


# In[9]:


split = data['train'].train_test_split(test_size = 0.3, seed = 42)
split


# Loading a pre-trained DistilBERT model is accomplished by specifying the model checkpoint as 'distilbert-base-cased'. The Hugging Face AutoTokenizer is then used to create a tokenizer for the chosen checkpoint. This tokenizer is crucial for converting raw text data into tokenized sequences compatible with the DistilBERT model.

# In[10]:


checkpoint = 'distilbert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# Tokenizing the dataset involves defining a tokenize_function that utilizes the Hugging Face tokenizer with truncation. The function processes batches of sentences from the dataset. The actual tokenization is applied using split.map, ensuring efficient batch-wise processing. This step is essential for converting raw text into tokenized representations suitable for training a machine learning model.

# In[11]:


def tokenize_function(batch):
    return tokenizer(batch['sentence'], truncation = True)

tokenized_dataset = split.map(tokenize_function, batched = True)


# In[12]:


model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 2)


# In[13]:


summary(model)


# Training parameters for the Hugging Face Trainer involves setting key values. The output_dir designates the directory for storing model checkpoints and outputs. The evaluation_strategy and save_strategy are both set to 'epoch,' indicating evaluation and checkpoint saving at each epoch's end. num_train_epochs is set to 1, determining the total training epochs. per_device_train_batch_size and per_device_eval_batch_size are set to 16 and 64, respectively, defining batch sizes for training and evaluation on each device.

# In[14]:


training_args = TrainingArguments(
    output_dir = 'training_dir',
    evaluation_strategy = 'epoch', 
    save_strategy = 'epoch', 
    num_train_epochs = 3, 
    per_device_train_batch_size = 16, 
    per_device_eval_batch_size = 64
    
)


# Defining a compute_metrics function for use with the Hugging Face Trainer, this function takes the output of the model predictions and true labels as input and computes accuracy and F1 score. The Trainer is then initialized with the model, training arguments, tokenized training and evaluation datasets, tokenizer, and the defined compute_metrics function. The training process is executed using the trainer.train() method.

# In[15]:


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


# Generating predictions on the test dataset with a pre-trained model. The `trainer.predict` method is employed to obtain predictions. then predicted labels are determined by selecting the indices with the highest values along the last axis of the prediction array. The confusion matrix is then calculated, using `confusion_matrix` function, comparing the true and predicted labels.

# In[16]:


predictions = trainer.predict(tokenized_dataset['test'])

predicted_labels = np.argmax(predictions.predictions, axis=-1)

true_labels = predictions.label_ids

conf_matrix = confusion_matrix(true_labels, predicted_labels)

print("Confusion Matrix:")
print(conf_matrix)


# In[22]:


plt.figure(figsize=(2, 2))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:




