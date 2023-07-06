import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import utilities as utl
import csv
import statistics



users = pd.read_csv("ml-100k/u.user", sep='|', header=None, engine='python', encoding='latin-1')

users.columns = ['Index', 'Age', 'Gender', 'Occupation', 'Zip code']

df = pd.read_csv("ml-100k/u.data", sep='	', header=None, engine='python', encoding='latin-1')
df.columns = ['uIndex', 'mIndex', 'rating', 'timestamp']

ratings=0.0
average=0.0
counter=0.0
rate_arr = []
for i in users['Index']:
    
    for j in range(0,99999):
        if i==df['uIndex'][j]:
            counter+=1
            ratings+=df['rating'][j]
            rate_arr.append(df['rating'][j])
    average=ratings/counter
    
    print(average)
    print(rate_arr)
    print(statistics.stdev(rate_arr))
    counter=0.0
    ratings=0.0
    average=0.0
    rate_arr=[]
