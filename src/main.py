import pandas as pd
import numpy as np

PATH = "data/caixas.csv"
DIMENSIONS= (600, 260, 244)

def readData(path):
    df = pd.read_csv(path)
    return df


df = readData(PATH)
valid = []
#print (df.describe())
for i in range(len(df)):
    line = [df.iloc[i]['largura'], df.iloc[i]['comprimento'], df.iloc[i]['altura']]
    line.sort(reverse=True)
    flag = True
    for j in range(len(DIMENSIONS)):
        if line[j] > DIMENSIONS[j]:
            flag = False
    if flag == True:
        valid.append(True)
    else:
        valid.append(False)
preprocessed = df[valid].copy()
preprocessed.loc[:, "volume"] = (preprocessed["largura"] * preprocessed["comprimento"] * preprocessed["altura"])/1e+6 #cm³ to m³
preprocessed.sort_values('volume', inplace=True)
print(preprocessed.head())
