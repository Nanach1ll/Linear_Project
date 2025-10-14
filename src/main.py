#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter
import re


# In[ ]:


file_path = 'roblox_games_data.csv'
# Load the latest version
df = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "databitio/roblox-games-data",
    file_path
)


# In[ ]:


df.head()
df.drop(["Date Created" , "Server Size" , "Last Updated" , "Date" , "Unnamed: 0" ],axis=1 , inplace=True)
df["Title"] = df["Title"].apply(lambda x: re.sub(r"^\[.*?\]", "", x).strip())

df.drop_duplicates(subset="Title", inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.sample(5)


# In[ ]:


df.shape


# In[ ]:


df.dtypes


# In[ ]:


df.info()


# In[ ]:


df.describe(include="all")


# In[ ]:


df["Genre"].unique()


# In[ ]:


df["Category"].unique()


# In[ ]:


df["text"] = df["Genre"].astype(str) + " " + df["Title"].astype(str) + " " + df["Category"].astype(str)
# รวม cols ที่เป็น text เป็นข้อความ เพราะ TF-IDF มันเหมาะกับงานพวก NLP
df.head()


# # เริ่ม ทำ

# In[ ]:


#ทำ tf-idf
import math
from collections import Counter, defaultdict

def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    # ลบ emoji + special chars, เหลือเฉพาะตัวอักษร a-z และตัวเลข
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # ลบช่องว่างซ้ำ
    text = re.sub(r'\s+', ' ', text).strip()
    # lowercase
    text = text.lower()
    return text


# Tokenize เป็นคำเล็ก ๆ
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

def tokenize(text):
    text = clean_text(text)
    return [stemmer.stem(w) for w in text.split()]

# TF: Term Frequency ของแต่ละคำในเอกสาร
def compute_tf(text):
    words = tokenize(text)
    return Counter(words)

# IDF: Inverse Document Frequency ของแต่ละคำใน corpus
def compute_idf(df, text_columns=['Title','Description']):
    N = len(df)
    idf = {}
    doc_freq = defaultdict(int)

    for i, row in df.iterrows():
        words_in_doc = set()
        for col in text_columns:
            text = str(row.get(col, ''))
            words_in_doc.update(tokenize(text))
        for word in words_in_doc:
            doc_freq[word] += 1

    for word, df_count in doc_freq.items():
        idf[word] = math.log((N + 1) / (df_count + 1)) + 1
    return idf


# TF-IDF vector
def compute_tfidf(text, idf):
    tf = compute_tf(text)
    tfidf = {word: tf[word] * idf.get(word, 0.0) for word in tf}
    return tfidf


# In[ ]:


def cosine_similarity(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([v**2 for v in vec1.values()])
    sum2 = sum([v**2 for v in vec2.values()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    return numerator / denominator if denominator else 0.0


# In[ ]:


def text_similarity(row, query, idf, title_weight=3):
    title_text = str(row['Title'])
    desc_text = str(row.get('Description',''))

    # duplicate title words
    combined = (" ".join([title_text]*title_weight)) + " " + desc_text

    tfidf_doc = compute_tfidf(combined, idf)
    tfidf_query = compute_tfidf(query, idf)
    return cosine_similarity(tfidf_doc, tfidf_query)

def compute_features_for_query(df, query):
    idf = compute_idf(df, text_columns=['Title','Description'])  # ใช้ Title + Description ก็ได้
    df['text_similarity'] = df.apply(lambda r: text_similarity(r, query, idf), axis=1)
    return df


# In[ ]:


def parse_number(s):
    if not isinstance(s, str):
        return float(s)
    s = s.replace('+','').replace(',','').strip()
    if s.endswith('M'):
        return float(s[:-1]) * 1_000_000
    elif s.endswith('K'):
        return float(s[:-1]) * 1_000
    try:
        return float(s)
    except:
        return 0.0


def train_linear_regression(X, y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add bias
    beta = np.linalg.pinv(X_b) @ y           # ใช้ pseudo-inverse
    return beta


def predict_linear_regression(X, beta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return X_b @ beta


# In[ ]:


# แปลงคอลัมน์
df['Total Visits_num'] = df['Total Visits'].apply(parse_number)
df['Favorites_num'] = df['Favorites'].apply(parse_number)
df['Active Users_num'] = df['Active Users'].apply(parse_number)

# normalize
df['visit_ratio_norm'] = df['Total Visits_num'] / df['Total Visits_num'].max()
df['fav_ratio_norm'] = df['Favorites_num'] / df['Favorites_num'].max()
df['active_ratio_norm'] = df['Active Users_num'] / df['Active Users_num'].max()

# สร้าง mapping ของ genre เป็น score
genre_mapping = {
    'Building': 0.6,
    'All Genres': 0.5,
    'Adventure': 0.7,
    'Fighting': 0.8,
    'RPG': 0.9,
    'Military': 0.7,
    'Town and City': 0.6,
    'Horror': 0.8,
    'FPS': 0.85,
    'Comedy': 0.5,
    'Naval': 0.6,
    'Sports': 0.7,
    'Sci-Fi': 0.8
}

df['genre_score'] = df['Genre'].map(genre_mapping).fillna(0.5)


# In[ ]:


def search_games(df, query, top=10):
    # 1. คำนวณ IDF ครอบคลุม Title + Description
    idf = compute_idf(df, text_columns=['Title','Description'])

    # 2. TF-IDF similarity
    df['text_similarity'] = df.apply(lambda r: text_similarity(r, query, idf, title_weight=3), axis=1)

    # 3. กรองเฉพาะแถวที่ text_similarity > 0
    df_filtered = df[(df['text_similarity'] > 0)].copy()

    # 4. สร้าง X matrix สำหรับ regression
    X_train = df_filtered[['text_similarity','visit_ratio_norm','fav_ratio_norm','genre_score']].values

    # 5. สร้าง target (ถ้าไม่มี user rating จริง)
    if 'UserRating' not in df_filtered.columns:
        df_filtered['UserRating'] = 0.4 * df_filtered['visit_ratio_norm'] + 0.3 * df_filtered['fav_ratio_norm'] + 0.3 * np.random.rand(len(df_filtered))

    y_train = df_filtered['UserRating'].values

    # 6. Train Linear Regression
    beta = train_linear_regression(X_train, y_train)
    print("Learned coefficients (β):", beta)

    # 7. Predict final score
    df_filtered['final_score'] = predict_linear_regression(X_train, beta)

    # 8. Sort by final_score
    df_sorted = df_filtered.sort_values(by='final_score', ascending=False)
    return df_sorted[['Title','Creator','URL','text_similarity','visit_ratio_norm','fav_ratio_norm','genre_score','final_score']].head(top)

