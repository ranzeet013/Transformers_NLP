
# Learning NLP with Transformers

On a journey of learning Natural Language Processing (NLP) with Transformers in this comprehensive exploration. Transformers, introduced by Vaswani et al., have reshaped the NLP landscape with their powerful self-attention mechanisms, excelling in tasks like sentiment analysis and machine translation. This learning venture guides you through essential NLP concepts, leveraging key libraries and tools, and delving into advanced topics. The transformative power of the Transformers library, featuring pre-trained models like BERT and GPT, is harnessed for practical implementations, bringing efficiency and robustness to NLP tasks. Whether you're a novice or an enthusiast, this journey promises to deepen your understanding and proficiency in NLP with state-of-the-art techniques and hands-on experiences.

## Journey Overview

In the vast field of Natural Language Processing, endless possibilities await. Over the next 60 days, 
I will explore fundamental concepts, essential libraries, and engage in hands-on tasks. Progressing from introductory topics to advanced NLP, I will conclude the journey with various practical project. This structured guide ensures a comprehensive learning experience, making it suitable for learners at any stage of expertise.

**1.Day_1 Of Learning NLP with Transformers :**

- **Basics of Natural Language Preprocessing :** Learning the basics of Natural Language Processing (NLP) with Transformers involves delving into the foundational principles of this interdisciplinary field, spanning Linguistics, Computer Science, and Artificial Intelligence. This journey revolves around deciphering how computers interact with human language, specifically focusing on programming them to process and analyze extensive amounts of natural language data. Currently, I am immersing myself in the early stages of NLP learning, where I've explored and implemented fundamental concepts. Within this exploration, I've provided concise insights into various libraries, dependencies, and modules essential for NLP. The emphasis is on processing text, akin to tasks such as removing retweet text, hyperlinks, and hashtags. I share this information in the hope that it will inspire others to gain insights and embark on their own NLP journey.

**2.Day_2 Of Learning NLP with Transformers :**

- **Sentiment Analysis :** In the realm of Natural Language Processing (NLP), sentiment analysis plays a central role in evaluating emotional tones within textual data. I import essential libraries for data manipulation, visualization, and NLP to construct a robust sentiment analysis workflow using a pre-trained model from the Transformers library. Demonstrating its versatility, the pipeline analyzes sentiments in two sentences and processes a Twitter dataset, strategically filtering data. Sentiments are numerically mapped, and predictions are made, leading to comprehensive model evaluation with accuracy, a heatmap-infused confusion matrix, and metrics like F1 score and ROC AUC score. This narrative highlights the practical application of sentiment analysis and the efficacy of a pre-trained model across the workflow, from preprocessing to evaluation.

Link:
[Sentiment Analysis](https://github.com/ranzeet013/Transformers_NLP/tree/main/01.%20Transformers%20for%20NLP/01.%20Sentiment%20Analysis)

**3.Day_3 Of Learning NLP with Transformers :**
- **Text Generation :** In my exploration of text generation using Transformers, I employed a pre-trained model from the Transformers library. The provided text was a passage about the Book of Genesis, serving as input for the text generation pipeline. Leveraging the Transformer model's self-attention mechanisms, it demonstrated the ability to comprehend and generate coherent text by capturing contextual dependencies within input sequences. The code utilized the 'pipeline' module from the Transformers library for text generation, and the generated text output showcased the model's proficiency in producing contextually rich and relevant sequences. I also experimented with different prompts, highlighting the flexibility and application of transformer-based text generation in diverse contexts, including creative writing and natural language understanding. To enhance the formatting and readability of the generated text, I incorporated the 'textwrap' module. Overall, this code revealed the capabilities and adaptability of transformer-based models for effective text generation.

Link:
[Text Generation GPT-2](https://github.com/ranzeet013/Transformers_NLP/tree/main/01.%20Transformers%20for%20NLP/02.%20Text%20Generation%20GPT-2)

**4.Day_4 Of Learning NLP with Transformers :**
- **Masked Language Modeling :** In the exploration of language modeling, I utilized the Transformers library to perform Masked Language Modeling (MLM), a crucial task in natural language processing (NLP) and machine learning. The provided text, "Ukraine Aid Falters in Senate as Republicans Insist on Border Restrictions," served as input for the MLM pipeline. Masked Language Modeling involves randomly masking or replacing certain words in a text with a special token. The model is then trained to predict the original words based on the contextual information provided by the surrounding words, enhancing its understanding of language semantics. The 'pipeline' module from the Transformers library was employed for MLM, allowing the model to predict the masked words in the given text. The code demonstrated the application of MLM in understanding and predicting missing words in context, showcasing its significance in pre-training language models for various NLP applications.

Link:
[Masked Language Modeling](https://github.com/ranzeet013/Transformers_NLP/tree/main/01.%20Transformers%20for%20NLP/03.%20Masked%20Language%20Modeling)

**5.Day_5 Of Learning NLP with Transformers :**
- **Named Entity Recognition :** I utilized the Transformers library to implement Named Entity Recognition (NER). The script initializes with a set of sample sentences covering diverse topics. Using the NER pipeline provided by Transformers, with a specified aggregation strategy, the script applies NER to identify and classify entities such as names, organizations, locations, and dates within each sentence. Named Entity Recognition, a fundamental natural language processing (NLP) technique, aims to extract valuable information and structurally represent text by assigning predefined categories to entities. This script demonstrates the practical application of NER for various purposes, including information retrieval, question answering, and sentiment analysis, showcasing the effectiveness of Transformers in advancing language-based systems.

Link:
[Named Entity Recognition](https://github.com/ranzeet013/Transformers_NLP/tree/main/01.%20Transformers%20for%20NLP/04.%20Named%20Entity%20Recognition)

**6.Day_6 Of Learning NLP with Transformers :**
- **Text Summarization :** In this script, I've explored text summarization using the Transformers library. The script begins by providing a news article on Ukraine Aid in the U.S. Senate. Leveraging the summarization pipeline from Transformers, the script then applies text summarization techniques. Text summarization is a crucial natural language processing (NLP) approach aimed at distilling large volumes of text while preserving essential information. The script utilizes a transformer-based model to generate a concise summary of the provided news article, showcasing the capabilities of Transformers in automating the summarization process. The resulting summary offers a condensed representation of the article's key points, demonstrating the efficiency of transformer models in extracting meaningful content from extensive textual data.

Link:
[Text Summarization](https://github.com/ranzeet013/Transformers_NLP/tree/main/01.%20Transformers%20for%20NLP/05.%20Text%20Summarization)

**7.Day_7 Of Learning NLP with Transformers :**
- **Neural Machine Translation :** In my exploration of Natural Language Processing (NLP), I delved into the world of machine translation using Python and the Transformers library. Specifically, I utilized a script to implement Neural Machine Translation (NMT), a deep learning approach that automatically translates text between languages. The script showcased the power of NMT, leveraging neural networks like transformers to comprehend entire input sentences at once, capturing contextual nuances for more fluent translations. It started by tokenizing words using a RegexpTokenizer and loading English to Spanish translation data. The BLEU score, a crucial metric for translation quality, was calculated, and NLTK's RegexpTokenizer was employed for tokenizing Spanish translations. The script concluded with the creation of a translation pipeline using the Helsinki-NLP model for English to Swedish translation, highlighting the practical application and versatility of NMT in language translation tasks.

Link:
[Neural MAchine Translation](https://github.com/ranzeet013/Transformers_NLP/tree/main/01.%20Transformers%20for%20NLP/06.%20Neural%20Machine%20Translation)

**8.Day_8 Of Learning NLP with Transformers :**
- **Question Answering :** While exploring the NLP, I utilized the Transformers library to create a pipeline for question-answering (QA). The pipeline, named 'qa,' is configured to perform QA tasks using a pre-trained model. In the example, a context about J. Robert Oppenheimer, a renowned physicist, is provided. The 'qa' pipeline is then used to answer a specific question, 'Who was J. Robert Oppenheimer?' The model processes the context and generates an answer based on its understanding of the input text. This showcases the practical application of the Transformers library in extracting meaningful information from textual data through QA tasks.

Link:
[Question Answering](https://github.com/ranzeet013/Transformers_NLP/tree/main/01.%20Transformers%20for%20NLP/07.%20Question%20Answering)

**9.Day_9 Of Learning NLP with Transformers :**
- **Zero-shot Classification :** In the exploration of text classification, I utilized the Hugging Face Transformers library to implement the zero-shot classification pipeline, an integral component of natural language processing (NLP). The script begins by importing essential libraries, such as pandas, numpy, and the Hugging Face Transformers library, followed by the initialization of the 'zero-shot-classification' pipeline stored in the variable classifier. Demonstrating its practical application, the script showcases a text classification example with the input "The anime was awesome," employing candidate labels 'Negative' and 'Positive,' and printing the resulting classification. Furthermore, the script introduces a more extensive example featuring a descriptive paragraph related to the anime "Attack on Titan." Although the explicit result is not printed, the script applies the classification pipeline to this longer text with candidate labels 'Movie' and 'Anime.' 

Link:
[Zero-shot classification](https://github.com/ranzeet013/Transformers_NLP/tree/main/01.%20Transformers%20for%20NLP/08.%20Zero-Shot%20Classification)


**10.Day_10 Of Learning NLP with Transformers :**
- **Models and Tokenizers :** On this day, I immersed myself in the fundamentals of natural language processing, utilizing the Hugging Face Transformers library with a specific focus on a BERT-based model using the 'bert-base-uncased' checkpoint. The exploration involved a crucial dive into tokenization, encompassing the processes of breaking down text into meaningful units, tokenizing, converting tokens to IDs, and decoding them. Transitioning to models, I loaded a pre-trained BERT-based model for sequence classification, demonstrating its application by preparing inputs, generating outputs, and customizing it for specific labels. The day's journey culminated in the practical application of the model to a list of sentences, incorporating considerations for input padding and truncation in real-world NLP tasks. This hands-on experience provided me with valuable skills in text data processing and the effective utilization of pre-trained models for sequence classification within the NLP domain.

Link:
[Models and Tokenizers](https://github.com/ranzeet013/Transformers_NLP/tree/main/01.%20Transformers%20for%20NLP/09.%20Models%20and%20Tokenizers)

**11.Day_11 Of Learning NLP with Transformers :**
- **Fine Tuning Sentiment Analysis :** On this day, I delved into the intricacies of sentiment analysis through fine-tuning a BERT-based model using Hugging Face Transformers. Beginning with the 'rotten_tomatoes' dataset, I loaded and tokenized text data, gaining insights into the tokenizer's functionality. A `tokenize_function` facilitated batch processing for the entire dataset. Configuring a `TrainingArguments` object, I initiated training for one epoch on a sentiment analysis model, obtaining valuable insights into its architecture through TorchInfo. After training, I saved the model and tested its capabilities using the Transformers pipeline, successfully classifying sentiments in a sample sentence. This hands-on experience provided a practical understanding of model fine-tuning and its application in sentiment analysis tasks, showcasing the versatility of the Hugging Face Transformers library in natural language processing workflows.

Link:
[Fine Tuning Sentiment Analysis](https://github.com/ranzeet013/Transformers_NLP/tree/main/01.%20Transformers%20for%20NLP/10.%20Fine%20Tuning%20Sentiment%20Analysis)


**12.Day_12 Of Learning NLP with Transformers :**
- **Multiclass Sentiment Analysis :** In today's coding journey, I ventured into the intricate world of sentiment analysis.I choose a BERT-based model fine-tuned with the Hugging Face Transformers. I started with the 'Tweets.csv' dataset, where I loaded and tokenized text data, getting an insider's view into the tokenizer's mechanics. A function, `tokenize_function`, handled batch processing like a champ. With a sleek TrainingArguments setup, I initiated a one-epoch training session on a sentiment analysis model, uncovering valuable insights into its architecture through TorchInfo. Post-training, I preserved the model and put it to the test using the Transformers pipeline, effortlessly classifying sentiments in a sample sentence. This hands-on adventure not only provided me with a practical grasp of model fine-tuning but also highlighted the Hugging Face Transformers library's remarkable versatility in the realm of natural language processing workflows.

Link:
[Multiclass Sentiment Analysis](https://github.com/ranzeet013/Transformers_NLP/tree/main/01.%20Transformers%20for%20NLP/11.%20Sentiment%20Classification/01.%20Multiclass%20Sentiment%20Analysis)


**13.Day_13 Of Learning NLP with Transformers :**
- **Binary Sentiment Classification :** On the 13th day of my coding journey, I immersed myself in sentiment analysis, harnessing the capabilities of a pre-trained DistilBERT model through the Hugging Face Transformers library. I curated a machine learning environment to explore the 'Tweets.csv' dataset. Filtering out 'neutral' sentiments, I visualized the distribution of positive and negative sentiments. Mapping sentiments to numerical values, I tokenized the dataset with a savvy tokenize_function. The DistilBERT model, with its architecture unveiled through torchinfo, underwent training orchestrated by the Hugging Face Trainer. Key parameters, including epochs and batch sizes, were fine-tuned through TrainingArguments. Post-training, the model was preserved, and predictions on the test dataset revealed a confusion matrix, visualized using Seaborn, providing insights into the model's sentiment classification performance. This coding odyssey illuminated the seamless integration of Hugging Face Transformers into sentiment analysis workflows, unveiling the power of natural language processing.

Link:
[Binary Sentiment Classification](https://github.com/ranzeet013/Transformers_NLP/tree/main/01.%20Transformers%20for%20NLP/11.%20Sentiment%20Classification/02.%20Binary%20Sentiment%20Classification)

**14.Day_14 of Learning NLP with Transformers :**
- **AutoConfig for Sentiment Analysis :** Engaging in sentiment analysis, I harnessed the capabilities of the 'distilbert-base-cased' checkpoint for tokenization and subsequent model training, leveraging the transformative power of the Hugging Face Transformers library. The dataset, originating from airline tweets, underwent meticulous loading and refinement to retain essential columns. Visualizing sentiment distribution was accomplished through a judiciously crafted bar plot. The establishment of a target mapping and subsequent tokenization unfolded seamlessly, enhancing the dataset's readiness for analysis. Key training parameters were adeptly configured using Hugging Face's TrainingArguments, complemented by a custom metrics function geared towards evaluating accuracy and macro F1 score. The instantiation of the Trainer facilitated the streamlined training of the model. Subsequent sections delved into predictions on the test dataset, the computation of a confusion matrix, and the artful visualization of results using seaborn. 

Link:
[AutoConfig for Sentiment Analysis](https://github.com/ranzeet013/Transformers_NLP/tree/main/01.%20Transformers%20for%20NLP/12.%20AutoConfig%20for%20Sentiment%20Analysis)




