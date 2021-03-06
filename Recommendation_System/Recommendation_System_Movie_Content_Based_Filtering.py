# -*- coding: utf-8 -*-
"""Proyek Akhir_MLT 2(Content Based Filtering)

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UBc5RmYdxcSnIYmVGUSzcF1SR55WC7dU
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

link = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/links.csv')
rating = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/ratings.csv')
movie = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/movies.csv')

link

rating

movie

link.info()

rating.info()

movie.info()

"""## Data Preprocessing"""

import numpy as np
 
# Menggabungkan seluruh movieId pada kategori Film
movie_all = np.concatenate((
    link.movieId.unique(),
    movie.movieId.unique(),
    rating.movieId.unique()
))
 
# Mengurutkan data dan menghapus data yang sama
movie_all = np.sort(np.unique(movie_all))
 
print('Jumlah seluruh data Film berdasarkan movieId: ', len(movie_all))

"""### Mengetahui Jumlah Rating"""

# Menggabungkan file link, tag, movie, rating ke dalam dataframe movie_info 
movie_info = pd.merge(link, movie, on = 'movieId')
movie_info

rate_group = rating.drop(['userId', 'timestamp'], axis = 1)
rate_group = rate_group.groupby(['movieId']).mean()
rate_group

movies = pd.merge(rate_group, movie_info , on='movieId')
movies

"""### Cek missing value dengan fungsi isnull()"""

movies.isnull().sum()

movies.loc[(movies['tmdbId'].isnull())]

"""Drop baris dengan nilai tmbdId yang null"""

movie_clean = movies.dropna()
movie_clean

"""Drop baris yang memiliki genres sebagai "No Genres Listed""""

movie_fix = movie_clean[movie_clean["genres"].str.contains("(no genres listed)")==False]
movie_fix

movie_fix.describe()

"""Mengambil rating yang memiliki rating diatas 2.5/5.0"""

movie_fix = movie_fix[(movie_fix['rating']>=2.5)]
movie_fix.describe()

movie_fix.info()

movie_fix

"""Melakukan pengecekan bahwa sudah tidak ada lagi *missing variable*"""

movie_fix.isnull().sum()

ratings = movie_fix['rating'].tolist()
title = movie_fix['title'].tolist()
genres = movie_fix['genres'].tolist()
 
print(len(ratings))
print(len(title))
print(len(genres))

movie_new = pd.DataFrame({
    'rating': ratings,
    'title': title,
    'genres': genres
})
movie_new = movie_new.drop_duplicates(subset=['title'])
movie_new = movie_new.dropna()
movie_new

"""Setelah data bersih maka lanjut ke dalam proses filtering

## Content Based Filtering
"""

from sklearn.feature_extraction.text import TfidfVectorizer

# Inisialisasi TfidfVectorizer
tf = TfidfVectorizer()
tf.fit(movie_new['genres'])

tf.get_feature_names()

tfidf_matrix = tf.fit_transform(movie_new['genres']) 
 
# Melihat ukuran matrix tfidf
tfidf_matrix.shape

tfidf_matrix.todense()

pd.DataFrame(
    tfidf_matrix.todense(), 
    columns=tf.get_feature_names(),
    index=movie_new.title
).sample(20, axis=1).sample(10, axis=0)

from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim

# Membuat dataframe dari variabel cosine_sim dengan baris dan kolom berupa nama resto
cosine_sim_df = pd.DataFrame(cosine_sim, index=movie_new['title'], columns=movie_new['title'])
print('Shape:', cosine_sim_df.shape)
 
# Melihat similarity matrix pada setiap resto
cosine_sim_df.sample(5, axis=1).sample(10, axis=0)

"""## Rekomendasi Data"""

def movie_recommendations(title_movie, similarity_data=cosine_sim_df, items=movie_new[['title', 'genres']], k=10):

    index = similarity_data.loc[:,title_movie].to_numpy().argpartition(range(-1, -k, -1))
    
    # Mengambil data dengan similarity terbesar dari index yang ada
    closest = similarity_data.columns[index[-1:-(k+2):-1]]

    closest = closest.drop(title_movie, errors='ignore')
 
    return pd.DataFrame(closest).merge(items).head(k)

movie_recommendations('How to Train Your Dragon 2 (2014)')