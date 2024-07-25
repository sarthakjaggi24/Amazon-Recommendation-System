import cv2
import numpy as np
import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from PIL import Image

amazon_df=pd.read_csv("C:\\Users\\Sarthak Jaggi\\Downloads\\amazon_product.csv")
amazon_df.drop('id',axis=1,inplace=True)

stemmer=SnowballStemmer('english')

def tokenize_stem(text):
    tokens=nltk.word_tokenize(text.lower())
    stemmed=[stemmer.stem(word) for word in tokens]
    return " ".join(stemmed)

amazon_df['stemmed_tokens']=amazon_df.apply(lambda row:tokenize_stem(row['Title']+ " " +row['Description']),axis=1)

tfidfv=TfidfVectorizer(tokenizer=tokenize_stem)

def cosine_sim(text1, text2):
  matrix = tfidfv.fit_transform([text1, text2])
  similarity_scores = cosine_similarity(matrix)

  if len(similarity_scores) > 0:
    return similarity_scores[0][0]
  else:
    return 0

def calculate_similarity(row, stemmed_text=None):
  stemmed_product = row['stemmed_tokens']
  similarity = cosine_sim(stemmed_text, stemmed_product)
  if similarity > 0:
    return similarity
  else:
    return 0

def search_product(text):
    stemmed_text=tokenize_stem(text)
    amazon_df['similarity']=amazon_df['stemmed_tokens'].apply(lambda x:cosine_sim(stemmed_text,x))
    result=amazon_df.sort_values(by=['similarity'],ascending=False).head(10)[['Title','Description','Category']]
    return result


img=Image.open("C:\\Users\\Sarthak Jaggi\\Downloads\\amazon photo.png")
st.image(img,width=600)

st.title("Amazon Product Recommendation System")

text=st.text_input("Enter Product Name")
submit=st.button('Search')

if submit:
    result=search_product(text)
    st.write(result)
