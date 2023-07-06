import numpy as np
import pandas as pd

data = pd.read_csv('moviesDataset/ratings.csv')
current = 1

movie_titles_genre = pd.read_csv("moviesDataset/movies.csv")
data = data.merge(movie_titles_genre,on='movieId', how='left')

Average_ratings = pd.DataFrame(data.groupby('title')['rating'].mean())
Average_ratings['Total Ratings'] = pd.DataFrame(data.groupby('title')['rating'].count())
movie_user = data.pivot_table(index='userId',columns='title',values='rating')
correlations = movie_user.corrwith(movie_user['Toy Story (1995)'])
recommendation = pd.DataFrame(correlations,columns=['Correlation'])
recommendation.dropna(inplace=True)
recommendation = recommendation.join(Average_ratings['Total Ratings'])
correlations = movie_user.corrwith(movie_user['Toy Story (1995)'])
recommendation = pd.DataFrame(correlations,columns=['Correlation'])
recommendation.dropna(inplace=True)
recommendation = recommendation.join(Average_ratings['Total Ratings'])
recc = recommendation[recommendation['Total Ratings']>100].sort_values('Correlation',ascending=False).reset_index()
recc = recc.merge(movie_titles_genre,on='title', how='left')

print(recc.head(10))

def acceptanceProbability(correlation):
    acceptanceProbability.counter+=1
    #print(acceptanceProbability.counter)
    return pow(2.72, -correlation*acceptanceProbability.counter)
acceptanceProbability.counter=0
print(acceptanceProbability(recc["Correlation"]))
