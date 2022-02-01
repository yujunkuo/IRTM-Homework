#!/usr/bin/env python
# coding: utf-8

# # NB Classifier

# ## Import

# In[1]:


import re
import math
import numpy as np
import pandas as pd
from os import listdir
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


# In[2]:


# Stemmer
STEMMER = PorterStemmer()

# Stop words
STOP_WORDS = stopwords.words("english")

# Corpus file path
CORPUS_FILE_PATH = "./data/IRTM/"


# ## Load Data

# In[3]:


# Load documents
files = listdir(CORPUS_FILE_PATH)
files = [f for f in files if f[0] != "."]

# Sort by file name
files.sort(key=lambda x: int(x[:-4]))

# Initialize corpus list: [[id, document], ...]
corpus = list()

# Read files
for file in files:
    with open(CORPUS_FILE_PATH + file, "r") as f:
        document_id = str(file)[:-4]
        document = f.read()
        corpus.append([document_id, document])

# corpus -> training + testing


# In[4]:


labels = dict()
with open("./data/training.txt", "r") as f:
    for line in f:
        data = line.split(" ")
        label = data[0]
        doc_list = data[1:-1]
        labels[label] = doc_list

# labels


# ## Training Data

# In[5]:


training_corpus = list()
for label in labels:
    for doc_id in labels[label]:
        training_corpus.append(corpus[int(doc_id)-1] + [label])


# In[6]:


training_df = pd.DataFrame(training_corpus)
training_df.columns = ["doc_id", "document", "label"]
training_df = training_df.astype({"doc_id": "int", "label": "int"})
training_df = training_df.sort_values(by="doc_id")
training_df = training_df.reset_index(drop = True)


# In[7]:


print(training_df.shape)
# training_df.head()


# ## Test Data

# In[8]:


test_corpus = list()
for doc in corpus:
    doc_id = doc[0]
    if int(doc_id) not in list(training_df["doc_id"]):
        test_corpus.append(corpus[int(doc_id)-1] + [None])


# In[9]:


test_df = pd.DataFrame(test_corpus)
test_df.columns = ["doc_id", "document", "label"]
test_df = test_df.astype({"doc_id": "int"})
test_df = test_df.sort_values(by="doc_id")
test_df = test_df.reset_index(drop = True)


# In[10]:


print(test_df.shape)
# test_df.head()


# ## Preprocessing

# In[11]:


def doc_preprocessing(doc: str) -> list:
    # Clean and lowercasing
    doc = doc.replace("\s+"," ").replace("\n", " ").replace("\r\n", " ")
    
    punc = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    digital = '0123456789'
    
    for token in punc:
        doc = doc.replace(token,"")
    for token in digital:
        doc = doc.replace(token,"")
        
    doc = re.sub(r"[^\w\s]", "", doc)
    doc = doc.lower()
    # Tokenization
    words = doc.split(" ")
    words = [word.strip() for word in words]
    # Porter's stemming
    stemming = [STEMMER.stem(word) for word in words]
    # Stop words removal
    token_list = [word for word in stemming if word and word not in STOP_WORDS]
    return token_list


# In[12]:


def get_tf(corpus: list):
    # Term frequency list
    # A list which store the TF dictionary of each document
    tf_list = list()
    
    # Iterate the corpus
    for document in corpus:
        
        # Words of each document
        document_word_list = doc_preprocessing(document)
        
        # Unique words and its frequency of each document
        tf = dict()
        for word in document_word_list:
            if word in tf:
                tf[word] += 1
            else:
                tf[word] = 1
        # Add this dictionary into the global list
        tf_list.append(tf)
    
    return tf_list


# In[13]:


training_df["tf"] = get_tf(training_df["document"])  # 取得 TF
training_df = training_df[["doc_id", "document", "tf", "label"]]  # 重新排列欄位順序
test_df["tf"] = get_tf(test_df["document"])  # 取得 TF
test_df = test_df[["doc_id", "document", "tf", "label"]]  # 重新排列欄位順序


# In[14]:


print(training_df.shape)
# training_df.head()


# In[15]:


print(test_df.shape)
# test_df.head()


# ## Feature Selection

# In[16]:


def extract_vocabulary(token_lists) -> list:
    return {token for token_list in token_lists for token in token_list}


# In[17]:


def chi_square(C: dict, D: pd.DataFrame):
    vocabulary = extract_vocabulary(D.tf)
    N = len(D)
    chi2 = dict()
    count = 0
    for term in vocabulary:
        chi2_term = 0
        matrix = dict()
        matrix["tp"] = D[D["tf"].apply(lambda x: term in x)]
        matrix["ta"] = D[D["tf"].apply(lambda x: term not in x)]
        for c in C:
            matrix["cp"] = D[D["label"] == int(c)]
            matrix["ca"] = D[D["label"] != int(c)]
            matrix["tp_cp"] = len(matrix["tp"][matrix["tp"]["label"] == int(c)])
            matrix["tp_ca"] = len(matrix["tp"][matrix["tp"]["label"] != int(c)])
            matrix["ta_cp"] = len(matrix["ta"][matrix["ta"]["label"] == int(c)])
            matrix["ta_ca"] = len(matrix["ta"][matrix["ta"]["label"] != int(c)])
            chi2_class = 0
            for i in ["tp", "ta"]:
                for j in ["cp", "ca"]:
                    E = len(matrix[i]) * len(matrix[j]) / N
                    chi2_class += ((matrix[f"{i}_{j}"] - E)**2) / E
            chi2_term += chi2_class
        chi2[term] = chi2_term
        count += 1
        if count % 500 == 0:
            print(f"Finish: {count}/{len(vocabulary)}")
    # 最終需要的 500 字
    vocabulary = sorted(chi2, key=chi2.get, reverse=True)[:500] 
    return vocabulary 


# In[18]:


vocabulary = extract_vocabulary(training_df.tf)
print(f"Vocabulary size before feature selection: {len(vocabulary)}")


# In[19]:


vocabulary = chi_square(labels, training_df)
print(f"Vocabulary size after feature selection: {len(vocabulary)}")


# ## NB Model

# In[20]:


def train_multinominal_nb(C: dict, D: pd.DataFrame, vocabulary):
    n_docs = len(D)
    prior = dict()
    cond_prob = {term: dict() for term in vocabulary}
    
    for c in C:
        n_class_docs = len(C[c])
        class_docs = D[D["label"] == int(c)]
        tct = dict()
        # 事前機率
        prior[c] = n_class_docs / n_docs
        # 事後機率
        for term in vocabulary:
            tokens_of_term = 0
            for tf in class_docs["tf"]:
                if term in tf:
                    tokens_of_term += tf[term]
            tct[term] = tokens_of_term
        for term in vocabulary:
            cond_prob[term][c] = (tct[term]+1) / (sum(tct.values())+len(vocabulary))
            
    return vocabulary, prior, cond_prob


# In[21]:


def apply_multinomial_nb(document, C, vocabulary, prior, cond_prob):
    tf = document["tf"]
    score = dict()
    
    for c in C:
        score[c] = math.log2(prior[c])
        for term in tf:
            if term in vocabulary:
                score[c] += (math.log2(cond_prob[term][c]))*(tf[term])
                # Bug: 因為取 log 是相加，因此出現多次時不應該用次方算，應該用相乘才對
            
    return max(score, key=score.get)


# ## Training and Test

# In[22]:


vocabulary, prior, cond_prob = train_multinominal_nb(labels, training_df, vocabulary)


# In[23]:


test_df["label"] = test_df.apply(
    apply_multinomial_nb, C=labels, vocabulary=vocabulary, prior=prior, cond_prob=cond_prob, axis=1)


# In[24]:


# test_df.head()


# ## Generate Output

# In[25]:


output_df = test_df[["doc_id", "label"]]
output_df.columns = ["Id", "Value"]


# In[26]:


print(output_df.shape)
# output_df.head()


# In[27]:


output_df.to_csv("./output/output.csv", index=False)

