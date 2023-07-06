from typing import ItemsView
import pandas as pd
import numpy as np
from zipfile import ZipFile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt
import statistics
import seaborn as sb
from scipy.stats import norm
import math
import os
import csv 


ratings_file = "ratingData1.csv"




df = pd.read_csv(ratings_file)
df.columns = ['uIndex', 'mIndex', 'rating', 'timestamp']


    
    

user_ids = df['uIndex'].unique().tolist()
movie_ids = df['mIndex'].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
df["user"] = df["uIndex"].map(user2user_encoded)
df["movie"] = df["mIndex"].map(movie2movie_encoded)

num_users = len(user2user_encoded)
num_movies = len(movie_encoded2movie)
df['rating'] = df["rating"].values.astype(np.float32)

min_rating = min(df["rating"])
max_rating = max(df["rating"])
print(
    "Number of users: {}, Number of Movies: {}, Min rating: {}, Max rating: {}".format(num_users, num_movies, min_rating, max_rating))
df = df.sample(frac=1, random_state=42)
x = df[["user", "movie"]].values
# Normalize the targets between 0 and 1. Makes it easy to train.
y = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
# Assuming training on 90% of the data and validating on 10%.
train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:],
)
EMBEDDING_SIZE = 50


class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.movie_embedding = layers.Embedding(
            num_movies,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.movie_bias = layers.Embedding(num_movies, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
        # Add all the components (including bias)
        x = dot_user_movie + user_bias + movie_bias
        # The sigmoid activation forces the rating to between 0 and 1
        return tf.nn.sigmoid(x)



def predict_rating(ratings_arr, item_ratings_arr):
  avg1 = sum(ratings_arr)/len(ratings_arr)     
  avg2 = sum(item_ratings_arr)/len(item_ratings_arr)
  
  standard_deviation = np.std(ratings_arr)
  noise = np.random.normal(0,1,1)
  omega = avg1+(standard_deviation*avg2)+noise
  print(noise)
  print(omega)
  value = max(min(round(omega[0]),5),1)
  return value


def selectionProbability(rank):
    return math.e**(-2*rank)

model = RecommenderNet(num_users, num_movies, EMBEDDING_SIZE)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(lr=0.001)
)




history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=64,
    epochs=5,
    verbose=1,
    validation_data=(x_val, y_val),
)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()

def find_index(input, file): 
    o = open(file, 'r') 
    myData = csv.reader(o) 
    index = 0 
    for row in myData:
      #print row
      if row[0] == input: 
        return index 
      else : index+=1

genre_tracker = {"unknown": 0.0, "Action": 0.0, "Adventure": 0.0, "Animation": 0.0,
              "Children's": 0.0, "Comedy": 0.0, "Crime": 0.0, "Documentary": 0.0, "Drama": 0.0, "Fantasy": 0.0,
              "Film-Noir": 0.0, "Horror": 0.0, "Musical": 0.0, "Mystery": 0.0, "Romance": 0.0, "Sci-Fi": 0.0,
              "Thriller": 0.0, "War": 0.0, "Western": 0.0}


movie_df = pd.read_csv("ml-100k/u.item", sep='|', header=None, engine='python', encoding='latin-1')
movie_df.columns = ["mIndex", "title", "release date", "video release date",
              "IMDb URL", "unknown", "Action", "Adventure", "Animation",
              "Childrens", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
              "Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
              "Thriller", "War", "Western"]

def getMoviesPerGenre():
    for i in range(0,99999):
            for z in range(0,1682):
                if df["mIndex"][i]==movie_df["movie id"][z]:
                    for l in genresList:
                        if movie_df[l][z]==1:
                            genre_tracker[l]+=1.0

def getNumGenres(csv,user_id):
    genreCounter = 0
    row_val = find_index()
    for l in genresList:
        if data[l][row_val]!=0:
            genreCounter+=1
    return genreCounter


with open('ratingData22.csv', 'w') as file:
    writer = csv.writer(file)
    for i in range(0,109428):
        writer.writerow([df["uIndex"][i], df["mIndex"][i], float(df["rating"][i]), df["timestamp"][i]])
    for x in range(943):
# Let us get a user and see the top recommendations.
        user_id = df["uIndex"].sample(1).iloc[0]
        movies_watched_by_user = df[df["uIndex"] == user_id]
        movies_not_watched = movie_df[
            ~movie_df["mIndex"].isin(movies_watched_by_user.mIndex.values)
        ]["mIndex"]
        movies_not_watched = list(
            set(movies_not_watched).intersection(set(movie2movie_encoded.keys()))
        )
        movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]
        user_encoder = user2user_encoded.get(user_id)
        user_movie_array = np.hstack(
            ([[user_encoder]] * len(movies_not_watched), movies_not_watched)
        )   
        ratings = model.predict(user_movie_array).flatten()
        top_ratings_indices = ratings.argsort()[-10:][::-1]
        recommended_movie_ids = [
            movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices
        ]

        print("Showing recommendations for user: {}".format(user_id))
        print("====" * 9)
        print("Movies with high ratings from user")
        print("----" * 8)
        top_movies_user = (
            movies_watched_by_user.sort_values(by="rating", ascending=False)
            .head(5)
            .mIndex.values
        )
        movie_df_rows = movie_df[movie_df["mIndex"].isin(top_movies_user)]
        ratings_array = []
        movie_ratings_array = []


        for j in range(0,109428):
            if df['uIndex'][j] == user_id:
                ratings_array.append(df['rating'][j])
        




        print("----" * 8)
        print("Top 10 movie recommendations")
        print("----" * 8)
        recommended_movies = movie_df[movie_df["mIndex"].isin(recommended_movie_ids)]
        for row in recommended_movies.itertuples():
            movie_ratings_array = []
            print(row.title, ":", row.Mystery)
            for j in range(0,109428):
                if df["mIndex"][j]==row.mIndex:
                    movie_ratings_array.append(df["rating"][j])
            print("predicted rating:"+str(predict_rating(ratings_array, movie_ratings_array)))
            writer.writerow([user_id, row.mIndex, predict_rating(ratings_array, movie_ratings_array), 888205088])
            data = pd.read_csv("userProfileGD3.csv")
            header = ["user id", "unknown", "Action", "Adventure", "Animation",
              "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
              "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
              "Thriller", "War", "Western"]
            genresList = ["unknown", "Action", "Adventure", "Animation",
              "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
              "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
              "Thriller", "War", "Western"]

            p = []
            for j in range(0,943):
                if data["user id"][j]==user_id:
                    for i in header:
                        p.append(data[header[i]][j])
            q = []
            #get genre distribution of recommended item
            #create array of recommended ItemsView
            #calculate genre distribution for all  
            
            for z in movie_df["mIndex"]:
                if row.mIndex==z:
                    row_val = find_index()
                    getMoviesPerGenre()
                    if row.unknown == 1:
                        data.loc[row_val, data["unknown"][row_val]] = genre_tracker["unknown"]/getNumGenres(,user_id)




