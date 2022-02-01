#!/usr/bin/env python
# coding: utf-8

# # HAC Clustering

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

# Doc Size
DOC_SIZE = 1095


# ## 文檔前處理函數

# In[3]:


def doc_preprocessing(doc: str) -> list:
    # Clean and lowercasing
    doc = doc.replace("\s+"," ").replace("\n", "").replace("\r\n", "")
    doc = re.sub(r"[^\w\s]", "", doc)
    doc = doc.lower()
    # Tokenization
    words = doc.split(" ")
    # Porter's stemming
    stemming = [STEMMER.stem(word) for word in words]
    # Stop words removal
    token_list = [word for word in stemming if word not in STOP_WORDS]
    return token_list


# ## 計算 TF 與 DF 函數

# In[4]:


def get_tf_and_df(corpus: list):
    # Term frequency list
    # A list which store the TF dictionary of each document
    tf_list = list()
    
    # Document frequency dictionary
    # A dictionary which store the DF of the corpus
    df_dict = dict()
    
    # Iterate the corpus
    for each in corpus:
        
        document_id, document = each
        
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
        tf_list.append([document_id, tf])
        
        # Calculate document frequency
        for term in tf:
            if term in df_dict:
                df_dict[term] += 1
            else:
                df_dict[term] = 1
                
    # Sort the DF dictionary
    df_dict = dict(sorted(df_dict.items(), key=lambda x: x[0]))
    
    return tf_list, df_dict


# ## 建立 Index Dictionary 函數

# In[5]:


def get_index_dict(df_dict: dict) -> dict:
    # Construct the Word-Index Mapping (Indexing)
    index_dict = dict()
    idx = 0
    for term in df_dict:
        index_dict[term] = idx
        idx += 1
    return index_dict  # (word: index)


# ## 計算 TF 與 TF-IDF 函數

# In[6]:


def get_tf_vector(tf_list, index_dict):
    # Initialize list to store TF vectors
    tf_vectors = list()
    # Iterate each TF list to generate TF vector
    for each in tf_list:
        document_id, tf_dict = each
        # Initialize vector with 0's
        tf_vector = np.array([0] * len(index_dict), dtype=float)
        # Store TF into vector
        for word in tf_dict:
            tf_vector[index_dict[word]] = tf_dict[word]
        # Store vector into list (with id mapping)
        tf_vectors.append([document_id, tf_vector])
    return tf_vectors


# In[7]:


def get_tf_idf_vector(tf_vectors, df_dict, index_dict):
    # Initialize vector with 0's
    idf_vector = np.array([0] * len(index_dict), dtype=float)
    # Store IDF into vector
    for word, df in df_dict.items():
        idf = math.log(len(tf_vectors) / df, 10)
        idf_vector[index_dict[word]] = idf
    # Calculate TF-IDF for all documents
    tf_idf_vectors = list()
    for tf_vector in tf_vectors:
        idx = tf_vector[0]
        tf_idf = tf_vector[1] * idf_vector
        tf_idf_unit = tf_idf / np.linalg.norm(tf_idf)
        tf_idf_vectors.append([idx, tf_idf_unit])
    return np.array(tf_idf_vectors)


# ## 計算 Cosine Similarity 函數

# In[8]:


def extract_vector(doc_id, doc_vectors):
    return doc_vectors[doc_id-1]

def cosine(doc_x, doc_y):
    vector_x = extract_vector(doc_x, doc_vectors)
    vector_y = extract_vector(doc_y, doc_vectors)
    cosine_sim = float(np.dot(vector_x, vector_y))
    return cosine_sim    


# ## 取得並儲存 Document vectors

# In[9]:


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

# Get TF and DF
# Term frequency list: [[id, {word: tf, ...}], ...]
# Document frequency dictionary: {word: df}
tf_list, df_dict = get_tf_and_df(corpus)

# Get Index Dictionary
# Index Dictionary: {Word: Index}
index_dict = get_index_dict(df_dict)

# Get TF vectors
tf_vectors = get_tf_vector(tf_list, index_dict)


# In[10]:


# Get TF-IDF vectors array
tf_idf_vectors = get_tf_idf_vector(tf_vectors, df_dict, index_dict)
doc_vectors = np.array([each[1] for each in tf_idf_vectors])
# Doc k at Index k-1 (ex. Doc1 at Index0)


# ## Clustering

# In[14]:


def get_max_similarity(C, I, DOC_SIZE):
    max_sim = -1
    doc_i, doc_m = -1, -1
    for i in range(DOC_SIZE):
        if I[i] != 1:
            continue
        for m in range(DOC_SIZE):
            if I[m] == 1 and i != m:
                if max_sim < C[i][m]:
                    max_sim = C[i][m]
                    doc_i, doc_m = i, m
    return doc_i, doc_m


# In[ ]:


def write_result(hac_dict, cluster_num):
    with open(f"./{cluster_num}.txt", "w") as f:
        for k, v in hac_dict.items():
            for doc_id in sorted(v):
                f.write(f"{doc_id+1}\n")
            f.write("\n")


# In[11]:


# Get cosine similarity array
C = np.array([[cosine(doc_x, doc_y) for doc_x in range(1, DOC_SIZE+1)] for doc_y in range(1, DOC_SIZE+1)])


# In[12]:


I = np.array([1]* DOC_SIZE)


# In[13]:


A = []


# In[15]:


for k in range(DOC_SIZE-1):
    i, m = get_max_similarity(C, I, DOC_SIZE)
    A.append([i ,m])
    for j in range(DOC_SIZE):
        C[i][j] = min(cosine(i, j), cosine(m, j))
        C[j][i] = min(cosine(j, i), cosine(j, m))
    I[m] = 0


# In[18]:


hac_dict = {str(i) : [i] for i in range(DOC_SIZE)}
for doc_i, doc_m in A:
    new_element = hac_dict[str(doc_m)]
    hac_dict.pop(str(doc_m))
    hac_dict[str(doc_i)] += new_element
    if len(hac_dict) == 20:
        write_result(hac_dict, 20)
    if len(hac_dict) == 13:
        write_result(hac_dict, 13)
    if len(hac_dict) == 8:
        write_result(hac_dict, 8)

