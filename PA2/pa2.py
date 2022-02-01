#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import math
import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from os import listdir


# ## 全域變數 Global variables

# In[2]:


### Global variables ###

# Stemmer
STEMMER = PorterStemmer()

# Stop words
STOP_WORDS = stopwords.words("english")

# Corpus file path
CORPUS_FILE_PATH = "./data/"

########################


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
    return tf_idf_vectors


# ## 計算 Cosine Similarity 函數

# In[38]:


def extract_vector(doc_id, index_dict):
    vector = np.array([0] * len(index_dict), dtype=float)
    with open(f"./output/doc{doc_id}.txt") as f:
        flag = 0
        for line in f:
            if flag > 1:
                idx, tf_idf = [x.strip() for x in re.split(r'\t+', line)]
                vector[int(idx)] = tf_idf
            flag += 1
    return vector

def cosine(doc_x, doc_y):
    vector_x = extract_vector(doc_x, index_dict)
    vector_y = extract_vector(doc_y, index_dict)
    cosine_sim = float(np.dot(vector_x, vector_y))
    return cosine_sim    


# ## Main

# In[8]:


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

# Get TF-IDF vectors
tf_idf_vectors = get_tf_idf_vector(tf_vectors, df_dict, index_dict)


# In[9]:


# Save the dictionary as a txt file
with open("./dictionary.txt", "w") as f:
    f.write("t_index\tterm\tdf\n")  # Head
    for key in df_dict:
        idx = index_dict[key]
        term = key
        df = df_dict[key]
        f.write(f"{idx}\t{term}\t{df}\n")


# In[13]:


# Save all the TF-IDF as txt file
for tf_idf_vector in tf_idf_vectors:
    doc_id, vector = tf_idf_vector
    terms_count = np.count_nonzero(vector)
    with open(f"./output/doc{doc_id}.txt", "w") as f:
        f.write(f"{terms_count}\n")  # Count the terms in the document
        f.write("t_index\ttf-idf\n")
        for i in range(len(vector)):
            if vector[i] != 0:
                f.write(f"{i}\t{vector[i]}\n")


# In[46]:


# Calculate and print the cosine similarity
doc_x, doc_y = 1, 2
print(f"Cosine similarity between doc{doc_x} and doc{doc_y} is {round(cosine(doc_x, doc_y), 6)}")

