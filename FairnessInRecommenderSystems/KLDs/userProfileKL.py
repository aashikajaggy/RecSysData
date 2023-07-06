import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import utilities as utl
import csv

users = pd.read_csv("ml-100k/u.user", sep='|', header=None, engine='python', encoding='latin-1')

# Columns describing user characteristics
users.columns = ['Index', 'Age', 'Gender', 'Occupation', 'Zip code']

df = pd.read_csv("ml-100k/u.data", sep='	', header=None, engine='python', encoding='latin-1')
df.columns = ['uIndex', 'mIndex', 'rating', 'timestamp']

movies = pd.read_csv("ml-100k/u.item", sep='|', header=None, engine='python', encoding='latin-1')
movies.columns = ["movie id", "movie title", "release date", "video release date",
              "IMDb URL", "unknown", "Action", "Adventure", "Animation",
              "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
              "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
              "Thriller", "War", "Western"]

header = ["user id", "unknown", "Action", "Adventure", "Animation",
              "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
              "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
              "Thriller", "War", "Western"]

genres = []

genresList = ["unknown", "Action", "Adventure", "Animation",
              "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
              "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
              "Thriller", "War", "Western"]
genre_dist = []
with open('userProfileGD7.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for i in users['Index']:
        for j in range(0,99999):
            if df['uIndex'][j]==i:
                for z in range(0,1682):
                    if df['mIndex'][j]==movies["movie id"][z]:
                        for l in movies.columns:
                            if movies[l][z]==1 and l not in genres:
                                genres.append(l)
                                print(l)
        counter=1
        total_genres=0
        genre_tracker = {"unknown": 0.0, "Action": 0.0, "Adventure": 0.0, "Animation": 0.0,
              "Children's": 0.0, "Comedy": 0.0, "Crime": 0.0, "Documentary": 0.0, "Drama": 0.0, "Fantasy": 0.0,
              "Film-Noir": 0.0, "Horror": 0.0, "Musical": 0.0, "Mystery": 0.0, "Romance": 0.0, "Sci-Fi": 0.0,
              "Thriller": 0.0, "War": 0.0, "Western": 0.0}
        val_arr = []
    
        '''      
        for i in range(0,len(genres)-1):
            if genres[i] != "":
                total_genres+=1
                for j in range(i+1,len(genres)-1):
                    if genres[i]==genres[j]:
                        counter+=1
                        genres[j]=""
                genre_tracker[genres[i]]=counter
                counter=0
                genres[i]=""
        '''

        for i in range(0,99999):
            for z in range(0,1682):
                if df["mIndex"][i]==movies["movie id"][z]:
                    for l in genresList:
                        if movies[l][z]==1:
                            genre_tracker[l]+=1.0

            # write the data
        genres = list(set(genres))
        val_arr = [i]
        for key, value in genre_tracker.items():
            val_arr.append(value/len(genres))
        writer.writerow(val_arr)
        genres = []
        val_arr = []

print(genre_tracker)
        


    
