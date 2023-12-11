
# Learning Natural Language Processing (NLP)

On a journey of learning Natural Language Processing (NLP) with Transformers in this comprehensive exploration. Transformers, introduced by Vaswani et al., have reshaped the NLP landscape with their powerful self-attention mechanisms, excelling in tasks like sentiment analysis and machine translation. This learning venture guides you through essential NLP concepts, leveraging key libraries and tools, and delving into advanced topics. The transformative power of the Transformers library, featuring pre-trained models like BERT and GPT, is harnessed for practical implementations, bringing efficiency and robustness to NLP tasks. Whether you're a novice or an enthusiast, this journey promises to deepen your understanding and proficiency in NLP with state-of-the-art techniques and hands-on experiences.

## Journey Overview

In the vast field of Natural Language Processing, endless possibilities await. Over the next 60 days, 
I will explore fundamental concepts, essential libraries, and engage in hands-on tasks. Progressing from introductory topics to advanced NLP, I will conclude the journey with various practical project. This structured guide ensures a comprehensive learning experience, making it suitable for learners at any stage of expertise.

**1.Day_1 Of Learning Natural Language Preprocessing :**

- **Basics of Natural Language Preprocessing :** Learning the basics of Natural Language Processing (NLP) with Transformers involves delving into the foundational principles of this interdisciplinary field, spanning Linguistics, Computer Science, and Artificial Intelligence. This journey revolves around deciphering how computers interact with human language, specifically focusing on programming them to process and analyze extensive amounts of natural language data. Currently, I am immersing myself in the early stages of NLP learning, where I've explored and implemented fundamental concepts. Within this exploration, I've provided concise insights into various libraries, dependencies, and modules essential for NLP. The emphasis is on processing text, akin to tasks such as removing retweet text, hyperlinks, and hashtags. I share this information in the hope that it will inspire others to gain insights and embark on their own NLP journey.

**2.Day_2 Of Learning Natural Language Preprocessing :**

- **Sentiment Analysis :** In the realm of Natural Language Processing (NLP), sentiment analysis plays a central role in evaluating emotional tones within textual data. I import essential libraries for data manipulation, visualization, and NLP to construct a robust sentiment analysis workflow using a pre-trained model from the Transformers library. Demonstrating its versatility, the pipeline analyzes sentiments in two sentences and processes a Twitter dataset, strategically filtering data. Sentiments are numerically mapped, and predictions are made, leading to comprehensive model evaluation with accuracy, a heatmap-infused confusion matrix, and metrics like F1 score and ROC AUC score. This narrative highlights the practical application of sentiment analysis and the efficacy of a pre-trained model across the workflow, from preprocessing to evaluation.

Link:
[Sentiment Analysis](https://github.com/ranzeet013/Transformers_NLP/tree/main/01.%20Transformers%20for%20NLP/01.%20Sentiment%20Analysis)

**3.Day_3 Of Learning Natural Language Preprocessing :**
- **Text Generation :** In my exploration of text generation using Transformers, I employed a pre-trained model from the Transformers library. The provided text was a passage about the Book of Genesis, serving as input for the text generation pipeline. Leveraging the Transformer model's self-attention mechanisms, it demonstrated the ability to comprehend and generate coherent text by capturing contextual dependencies within input sequences. The code utilized the 'pipeline' module from the Transformers library for text generation, and the generated text output showcased the model's proficiency in producing contextually rich and relevant sequences. I also experimented with different prompts, highlighting the flexibility and application of transformer-based text generation in diverse contexts, including creative writing and natural language understanding. To enhance the formatting and readability of the generated text, I incorporated the 'textwrap' module. Overall, this code revealed the capabilities and adaptability of transformer-based models for effective text generation.

Link:
[Text Generation GPT-2](https://github.com/ranzeet013/Transformers_NLP/tree/main/01.%20Transformers%20for%20NLP/02.%20Text%20Generation%20GPT-2)

**4.Day_4 Of Learning Natural Language Preprocessing :**
- **Masked Language Modeling :** In the exploration of language modeling, I utilized the Transformers library to perform Masked Language Modeling (MLM), a crucial task in natural language processing (NLP) and machine learning. The provided text, "Ukraine Aid Falters in Senate as Republicans Insist on Border Restrictions," served as input for the MLM pipeline. Masked Language Modeling involves randomly masking or replacing certain words in a text with a special token. The model is then trained to predict the original words based on the contextual information provided by the surrounding words, enhancing its understanding of language semantics. The 'pipeline' module from the Transformers library was employed for MLM, allowing the model to predict the masked words in the given text. The code demonstrated the application of MLM in understanding and predicting missing words in context, showcasing its significance in pre-training language models for various NLP applications.

Link:
[Masked Language Modeling](https://github.com/ranzeet013/Transformers_NLP/tree/main/01.%20Transformers%20for%20NLP/03.%20Masked%20Language%20Modeling)

**5.Day_5 Of Learning Natural Language Preprocessing :**
- **Named Entity Recognition :** I utilized the Transformers library to implement Named Entity Recognition (NER). The script initializes with a set of sample sentences covering diverse topics. Using the NER pipeline provided by Transformers, with a specified aggregation strategy, the script applies NER to identify and classify entities such as names, organizations, locations, and dates within each sentence. Named Entity Recognition, a fundamental natural language processing (NLP) technique, aims to extract valuable information and structurally represent text by assigning predefined categories to entities. This script demonstrates the practical application of NER for various purposes, including information retrieval, question answering, and sentiment analysis, showcasing the effectiveness of Transformers in advancing language-based systems.

Link:
[Named Entity Recognition](https://github.com/ranzeet013/Transformers_NLP/tree/main/01.%20Transformers%20for%20NLP/04.%20Named%20Entity%20Recognition)

**5.Day_5 Of Learning Natural Language Preprocessing :**
- **Text Summarization :** In this script, I've explored text summarization using the Transformers library. The script begins by providing a news article on Ukraine Aid in the U.S. Senate. Leveraging the summarization pipeline from Transformers, the script then applies text summarization techniques. Text summarization is a crucial natural language processing (NLP) approach aimed at distilling large volumes of text while preserving essential information. The script utilizes a transformer-based model to generate a concise summary of the provided news article, showcasing the capabilities of Transformers in automating the summarization process. The resulting summary offers a condensed representation of the article's key points, demonstrating the efficiency of transformer models in extracting meaningful content from extensive textual data.

Link:
[Text Summarization](https://github.com/ranzeet013/Transformers_NLP/tree/main/01.%20Transformers%20for%20NLP/05.%20Text%20Summarization)

