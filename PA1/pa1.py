# IRTM - Programming Assignment 1
# R10725018

# Load Document
import requests as rq
import re
r = rq.get("https://ceiba.ntu.edu.tw/course/35d27d/content/28.txt")
doc = r.text.replace("\r\n", "")
doc = re.sub(r"[^\w\s]", "", doc)

# Tokenization and Lowercasing
tokenization = [word.lower() for word in doc.split(" ")]

# Porter's Stemming
from nltk.stem import PorterStemmer
ps = PorterStemmer()
stemming = [ps.stem(word) for word in tokenization]

# Stopword Removal
from nltk.corpus import stopwords
result = [word for word in stemming if word not in stopwords.words("english")]

# Save the result as a txt file
with open("result.txt", "w") as f:
    for word in result:
        f.write(word + "\n")