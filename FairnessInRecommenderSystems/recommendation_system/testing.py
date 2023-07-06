import pandas as pd
import numpy as np
import csv 


data = pd.read_csv("minorityIndexFINALEE1.csv")

def find_index(inputted, file): 
    o = open(file, 'r') 
    myData = csv.reader(o) 
    index = 0 
    for row in myData:
      #print row
      if row[0] == inputted: 
        return index 
      else : 
        index+=1

row_val = find_index("86", "minorityIndexFINALEE1.csv")

print(find_index("86", "minorityIndexFINALEE1.csv"))

value = find_index("86", "minorityIndexFINALEE1.csv")

data.at[row_val, "movie id"] = 500

print(data["movie id"][row_val])
