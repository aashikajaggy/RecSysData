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
header = ["movie id", "minority index"]

fsum=0.0
msum=0.0
fcounter=0.0
mcounter=0.0
val_arr = []
mIdArray = []
with open('minorityIndexFINALEE1.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for j in range(0,99999):
        for l in range(0,943):
            if df["mIndex"][j] not in mIdArray and df['uIndex'][j] == users['Index'][l]:
                if users['Gender'][l] == 'F':
                    fsum+=df['rating'][j]
                    fcounter+=1
                if users['Gender'][l] == 'M':
                    msum+=df['rating'][j]
                    mcounter+=1
        for t in range(j+1,99999):
            if df["mIndex"][t] not in mIdArray and df['mIndex'][j]==df['mIndex'][t]:
                for l in range(0,943):
                    if df['uIndex'][t] == users['Index'][l]:
                        if users['Gender'][l] == 'F':
                            fsum+=df['rating'][t]
                            fcounter+=1
                        if users['Gender'][l] == 'M':
                            msum+=df['rating'][t]
                            mcounter+=1
        mIdArray.append(df['mIndex'][j])
        if fcounter>0 and mcounter>0:
            avgf=fsum/fcounter
            avgm=msum/mcounter
            percent_change=(avgm-avgf)/avgf
            val_arr=[df['mIndex'][j]]
            val_arr.append(percent_change)
        writer.writerow(val_arr)
        fsum=0.0
        msum=0.0
        mcounter=0.0
        fcounter=0.0
        val_arr=[]
        percent_change=0.0
