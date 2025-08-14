import pandas as pd
import numpy as np

PATH = "data/caixas.csv"
#comprimento, largura , altura
DIMENSIONS= (600, 260, 244)
VOLUME_CONTAINER = (600 * 260 * 244) / 1e+6 #cm³ to m³
print(f"Container dimensions: {DIMENSIONS}, Volume: {VOLUME_CONTAINER} m³")
def readData(path):
    df = pd.read_csv(path)
    return df


df = readData(PATH)
valid = []
#print (df.describe())
for i in range(len(df)):
    line = [df.iloc[i]['largura'], df.iloc[i]['comprimento'], df.iloc[i]['altura']]
    #caixa "rotacionada", de tal forma que a maior dimensão represente sempre o comprimento
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

quantidade = 0
for idx, row in preprocessed.iterrows():
    quantidade += int(row['quantidade'])
print(f"Total number of boxes: {quantidade}")

#Permutar todas as rotações possíveis para todas as 74 caixas validas em forma de lista de lista para cada linha há uma lista de 6 rotações possiveis (3!)
rotacoes = []
for idx, row in preprocessed.iterrows():
    largura = int(row['largura'])
    comprimento = int(row['comprimento'])
    altura = int(row['altura'])
    rotacoes_lista = [
        (largura, comprimento, altura),
        (largura, altura, comprimento),
        (comprimento, largura, altura),
        (comprimento, altura, largura),
        (altura, largura, comprimento),
        (altura, comprimento, largura)
    ]
    rotacoes.append(rotacoes_lista)

preprocessed['rotacoes'] = rotacoes
print(preprocessed.head())

#Testando varias abordagens
#First fit por volume decreasing 
container = []
volume_container = VOLUME_CONTAINER
for idx, row in preprocessed.sort_values('volume', ascending=False).iterrows():
    volume =row['volume']
    quantidade = int(row['quantidade'])
    while quantidade > 0 and volume <= volume_container:
        quantidade -= 1
        volume_container -= volume
        container.append((idx, volume))
        print(f"Added box {idx} with volume {volume}, remaining volume in container: {volume_container}")
print(len(container), "boxes fit in the container by first fit volume approach")

#First fit por volume crescente
container = []
volume_container = VOLUME_CONTAINER
for idx, row in preprocessed.sort_values('volume', ascending=True).iterrows():
    volume =row['volume']
    quantidade = int(row['quantidade'])
    while quantidade > 0 and volume <= volume_container:
        quantidade -= 1
        volume_container -= volume
        container.append((idx, volume))
        print(f"Added box {idx} with volume {volume}, remaining volume in container: {volume_container}")
print(len(container), "boxes fit in the container by first fit volume crescente approach. com volume = ", volume)

#Extreme points

#Wall building

#Hibrida extreme points + wall building

#Otimizar a ordem e a prioridade dos 74 tipos com GA usando o fitnees como EP, WB ou hibrido 


