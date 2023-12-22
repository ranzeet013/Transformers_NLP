#!/usr/bin/env python
# coding: utf-8

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


# In[10]:


target_map = {'positive':'1', 'negative':'0', 'neutral':'2'}
dataframe['target'] = dataframe['airline_sentiment'].map(target_map)
dataframe.head()


# In[11]:


data = load_dataset('csv', data_files = 'data.csv')
data


# In[12]:


split = data['train'].train_test_split(test_size = 0.3, seed = 42)
split


# Utilizing the 'distilbert-base-cased' checkpoint, to initializes a tokenizer using the Hugging Face `AutoTokenizer` class. The `tokenize_function` is defined to tokenize sentences in batches, employing the specified tokenizer and enabling truncation for longer sentences. The dataset is then tokenized using the `map` method with batch processing enabled, resulting in the `tokenized_dataset`.

# In[13]:


checkpoint = 'distilbert-base-cased'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(batch):
    return tokenizer(batch['sentence'], truncation = True)

tokenized_dataset = split.map(tokenize_function, batched = True)


# In[14]:


from transformers import AutoConfig

config = AutoConfig.from_pretrained(checkpoint)
config


# In[15]:


config.id2label


#  The id2label dictionary is constructed by swapping the keys and values of the target_map items, and the label2id attribute is set to the original target_map.

# In[16]:


config.id2label = {v:k for k, v in target_map.items()}
config.label2id = target_map


# In[17]:


model = AutoModelForSequenceClassification.from_pretrained(checkpoint, config = config)


# In[18]:


summary(model)


# Defining training parameters for the Hugging Face Trainer involves setting key values. The `output_dir` designates the directory for storing model checkpoints and outputs. Both `evaluation_strategy` and `save_strategy` are set to 'epoch,' indicating evaluation and checkpoint saving at each epoch's end. `num_train_epochs` is set to 1, determining the total training epochs. `per_device_train_batch_size` and `per_device_eval_batch_size` are set to 16 and 64, respectively, defining batch sizes for training and evaluation on each device.

# In[19]:


training_args = TrainingArguments(
    output_dir = 'training_dir',
    evaluation_strategy = 'epoch', 
    save_strategy = 'epoch', 
    num_train_epochs = 1, 
    per_device_train_batch_size = 16, 
    per_device_eval_batch_size = 64
    
)


# Defining a custom `compute_metrics` function, to calculates accuracy and macro F1 score based on model predictions and true labels. Initializung a Trainer with the model, training arguments, tokenized training and test datasets, tokenizer, and the specified metrics computation function. Subsequently, the training process is executed using the `trainer.train()` method, enabling the model to learn from the provided training dataset.

# In[20]:


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


# Making predictions on the test dataset using a trained model, to calculates a confusion matrix to evaluate classification performance. The predicted labels are determined, and the matrix is generated using scikit-learn's `confusion_matrix` function. Subsequently, the confusion matrix is visualized as a heatmap using seaborn, providing an intuitive representation of model performance in terms of true and predicted labels for negative and positive sentiments.

# In[21]:


predictions = trainer.predict(tokenized_dataset['test'])

predicted_labels = np.argmax(predictions.predictions, axis=-1)

true_labels = predictions.label_ids

conf_matrix = confusion_matrix(true_labels, predicted_labels)

print("Confusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(3, 3))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:




