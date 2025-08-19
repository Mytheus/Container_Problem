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

for rotacao in rotacoes:
    set(rotacao)  
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
class ExtremePoints:
    def __init__(self, container_dim):
        self.container = container_dim  # (L, W, H)
        self.points = [(0,0,0)]        # pontos livres
        self.boxes = []                # caixas já colocadas

    def fits(self, point, box):
        x, y, z = point
        L, W, H = box
        # Verifica se ultrapassa o container
        if x + L > self.container[0] or y + W > self.container[1] or z + H > self.container[2]:
            return False
        # Verifica colisão com caixas já colocadas
        for (px, py, pz), (bL, bW, bH) in self.boxes:
            if (x < px + bL and x + L > px and
                y < py + bW and y + W > py and
                z < pz + bH and z + H > pz):
                return False
        return True

    def update_points(self, point, box):
        # Adiciona novos pontos extremos gerados pela caixa colocada
        x, y, z = point
        L, W, H = box
        new_points = [
            (x + L, y, z),
            (x, y + W, z),
            (x, y, z + H)
        ]
        # Só adiciona pontos que cabem no container
        for px, py, pz in new_points:
            if px <= self.container[0] and py <= self.container[1] and pz <= self.container[2]:
                self.points.append((px, py, pz))
        # Remove o ponto onde colocamos a caixa
        if point in self.points:
            self.points.remove(point)

    def place_box(self, box):
        for point in self.points:
            if self.fits(point, box):
                self.boxes.append((point, box))
                self.update_points(point, box)
                return True
        return False

    def utilization(self):
        # Retorna o volume usado do container
        return sum(b[1][0]*b[1][1]*b[1][2] for b in self.boxes)

#Wall building
class WallBuilding:
    def __init__(self, container_dim):
        self.container = container_dim
        self.layers = []

    def build_layer(self, boxes):
        layer = []
        width_used = 0
        for box in boxes:
            if width_used + box[0] <= self.container[0]:
                layer.append(box)
                width_used += box[0]
        self.layers.append(layer)
#Hibrida extreme points + wall building

#Otimizar a ordem e a prioridade dos 74 tipos com GA usando o fitnees como EP, WB ou hibrido 
def melhor_rotacao(rotacoes):
    # Retorna a rotação com maior área de base (L * W)
    return max(rotacoes, key=lambda r: r[0] * r[1])

# Cria lista de cromossomos
cromossos = [
    (
        idx,
        row['volume'],
        melhor_rotacao(row['rotacoes']),  # retorna a tupla (L, W, H)
        row['quantidade']
    )
    for idx, row in preprocessed.iterrows()
]

print(cromossos[:5])

# Preparando lista de boxes para usar no GA ou nas heurísticas
# Cada item será a melhor rotação da caixa repetida 'quantidade' vezes
boxes = []
for idx, volume, rot, quantidade in cromossos:
    boxes.extend([rot] * int(quantidade))

def fitness(cromossomo, method='EP'):
    if method == 'EP':
        container = ExtremePoints(DIMENSIONS)
    elif method == 'WB':
        container = WallBuilding(DIMENSIONS)
    # Coloca as caixas na ordem do cromossomo
    for idx in cromossomo:
        box = boxes[idx]  # agora boxes[idx] existe
        if method == 'EP':
            container.place_box(box)
        elif method == 'WB':
            container.build_layer([box])
    if method == 'EP':
        return sum(b[1][0]*b[1][1]*b[1][2] for b in container.boxes)/VOLUME_CONTAINER
    elif method == 'WB':
        return sum(b[0]*b[1]*b[2] for layer in container.layers for b in layer)/VOLUME_CONTAINER
    
#AG 
import random

# --- Parâmetros do GA ---
POP_SIZE = 50          # Número de indivíduos na população
GENERATIONS = 100      # Número de gerações
ELITISM = 0.2          # Percentual de elite a ser mantida
MUTATION_RATE = 0.1    # Probabilidade de mutação
METHOD = 'EP'          # EP, WB ou 'Hibrido' se implementar

# --- Inicialização da população ---
# Cada indivíduo é uma permutação aleatória dos índices de boxes
def initialize_population(pop_size, n_boxes):
    population = []
    for _ in range(pop_size):
        cromossomo = list(range(n_boxes))
        random.shuffle(cromossomo)
        population.append(cromossomo)
    return population

# --- Avaliação do fitness ---
def evaluate_population(population, method=METHOD):
    fitness_values = []
    for crom in population:
        f = fitness(crom, method)
        fitness_values.append(f)
    return fitness_values

# --- Seleção por elitismo ---
def select_elite(population, fitness_values, elitism_rate=ELITISM):
    n_elite = max(1, int(len(population) * elitism_rate))
    # Ordena população pelo fitness (decrescente)
    sorted_pop = [x for _, x in sorted(zip(fitness_values, population), reverse=True)]
    return sorted_pop[:n_elite]

# --- Crossover (Order Crossover simples) ---
def crossover(parent1, parent2):
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    child = [None]*size
    child[a:b] = parent1[a:b]
    fill_pos = b
    for gene in parent2:
        if gene not in child:
            if fill_pos >= size:
                fill_pos = 0
            child[fill_pos] = gene
            fill_pos += 1
    return child

# --- Mutação (swap de dois genes) ---
def mutate(cromossomo, mutation_rate=MUTATION_RATE):
    if random.random() < mutation_rate:
        a, b = random.sample(range(len(cromossomo)), 2)
        cromossomo[a], cromossomo[b] = cromossomo[b], cromossomo[a]
    return cromossomo

# --- Geração da nova população ---
def next_generation(population, fitness_values):
    new_pop = select_elite(population, fitness_values)  # mantém a elite
    while len(new_pop) < len(population):
        # Seleciona dois pais aleatórios
        parent1, parent2 = random.choices(population, k=2)
        child = crossover(parent1, parent2)
        child = mutate(child)
        new_pop.append(child)
    return new_pop

# --- Loop principal do GA ---
def run_ga(boxes, generations=GENERATIONS):
    population = initialize_population(POP_SIZE, len(boxes))
    best_solution = None
    best_fitness = -1

    for gen in range(generations):
        fitness_values = evaluate_population(population)
        max_fitness = max(fitness_values)
        if max_fitness > best_fitness:
            best_fitness = max_fitness
            best_solution = population[fitness_values.index(max_fitness)]
        print(f"Generation {gen+1}: Best fitness = {max_fitness:.4f}")
        population = next_generation(population, fitness_values)

    return best_solution, best_fitness

best_cromossomo, best_fit = run_ga(boxes)
print("\nMelhor solução encontrada:")
print(best_cromossomo)
print(f"Fitness: {best_fit:.4f}")
