# CI 1 : Implement Union, Intersection, Complement and Difference operations on fuzzy sets. Also create fuzzy relations by Cartesian product of any two fuzzy sets and perform max-min composition on any two # fuzzy relations.

import numpy as np
#1
A = np.array([0.2, 0.4, 0.6, 0.8])
B = np.array([0.5, 0.3, 0.7, 0.9])

#2
def union(A, B): return np.maximum(A, B)
def intersection(A, B): return np.minimum(A, B)
def compliment(A): return 1 - A
def difference(A, B): return np.minimum(A, 1 - B)
print("Union :\n", union(A, B))
print("Intersection :\n", intersection(A, B))
print("Compliment :\n", compliment(A))
print("Difference :\n", difference(A, B))

#3
P = np.array([0.2, 0.5, 0.8])
Q = np.array([0.4, 0.6, 0.9])
R = np.array([0.3, 0.7, 1.0])

#4
def cartesian_product(A, B):
    return np.array([[min(a, b) for b in B]for a in A])

#5
R1 = cartesian_product(P, Q)
R2 = cartesian_product(Q, R)
print("P x Q :\n", R1)
print("Q x R :\n", R2)

#6
def max_min(R, S):
    rows, mid = R.shape
    cols = S.shape[1]
    result = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            result[i][j] = max(min(R[i][k], S[k][j])for k in range(mid))
    return result

print("Composition :\n", max_min(R1, R2))


# CI 2 : Optimization of genetic algorithm parameter in hybrid genetic algorithm-neural network modelling: Application to spray drying of coconut milk.

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("iris.csv")

x = data.iloc[:, :-1]
y = LabelEncoder().fit_transform(data.iloc[:, -1])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Fitness function
def fitness(p):
    model = MLPClassifier(hidden_layer_sizes=(p[0],),
                          learning_rate_init=p[1],
                          max_iter=200)
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)

# Initial population
pop = [[np.random.randint(5,50),
        np.random.uniform(0.001,0.1)] for _ in range(5)]

# Genetic Algorithm
for _ in range(10):

    pop = sorted(pop, key=fitness, reverse=True)

    p1, p2 = pop[0], pop[1]

    child = [(p1[0]+p2[0])//2,
             (p1[1]+p2[1])/2]

    pop[-1] = child

# Best solution
best = pop[0]

print("Best Parameters:", best)


# CI 3 : Implementation of Clonal selection algorithm using Python

import numpy as np

def f(x): return (x-2)**2
def aff(f): return 1/(1+f)

pop = np.random.uniform(-10, 10, 20)

for _ in range(50):
    fit = np.array([f(x) for x in pop])
    af = np.array([aff(fi) for fi in fit])

    sel = pop[np.argsort(af)[-10:]]
    clones = []

    for s in sel:
        c = np.repeat(s, 5)
        c = c + np.random.normal(0, 0.5 * (1 - aff(f(s))), 5)
        clones.extend(c)

    clones = np.array(clones)
    best = pop[np.argsort([f(x) for x in clones][:15])]
    new = np.random.uniform(-10, 10, 5)
    pop = np.concatenate([best, new])

best = pop[np.argmin([f(x) for x in pop])]
print("Best x :", best)


# CI 4 : Implement DEAP (Distributed Evolutionary Algorithms) using Python

import random

# Fitness function (maximize x^2)
def fitness(ind):
    return ind[0] ** 2

# Create random individual
def create_individual():
    return [random.uniform(-10, 10)]

# Create population
def create_population(size):
    return [create_individual() for _ in range(size)]

# Selection (pick best from 3 random)
def select(pop):
    return max(random.sample(pop, 3), key=fitness)

# Crossover (average of parents)
def crossover(p1, p2):
    return [(p1[0] + p2[0]) / 2]

# Mutation (small change)
def mutate(ind):
    if random.random() < 0.1:
        ind[0] += random.uniform(-1, 1)
    return ind

# Evolve one island
def evolve(pop):
    new_pop = []
    for _ in range(len(pop)):
        p1 = select(pop)
        p2 = select(pop)
        child = crossover(p1, p2)
        child = mutate(child)
        new_pop.append(child)
    return new_pop

# Migration (share best solutions)
def migrate(islands):
    for i in range(len(islands)):
        best = max(islands[i], key=fitness)
        next_island = (i + 1) % len(islands)
        islands[next_island][0] = best # replace worst/simple replace
    return islands

# MAIN
# Create islands
islands = [create_population(5) for _ in range(3)]

# Run for iterations
for gen in range(10):

    # Evolve each island
    islands = [evolve(pop) for pop in islands]

    # Migration every 2 generations
    if gen % 2 == 0:
        islands = migrate(islands)

    # Print best of each island
    print(f"\nGeneration {gen+1}")

    for i, pop in enumerate(islands):
        best = max(pop, key=fitness)

        print(f"Island {i}: Best = {best[0]:.2f}, "
              f"Fitness = {fitness(best):.2f}")

# Final best
all_ind = [ind for pop in islands for ind in pop]

best = max(all_ind, key=fitness)

print("\nFinal Best Solution:")
print("x =", best[0])
print("Fitness =", fitness(best))


# CI 5 : Implement Ant colony optimization by solving the Traveling salesman problem using python Problem statement- A salesman needs to visit a set of cities exactly once and return to the original city. The # task is to find the shortest possible route that the salesman can take to visit all the cities and return to the starting city.

import numpy as np

d=np.array([[0,2,9,10],
            [1,0,6,4],
            [15,7,0,8],
            [6,3,12,0]])

n=len(d)
p=np.ones((n,n))

def f(r):
    return sum(d[r[i]][r[i+1]] for i in range(n-1))+d[r[-1]][r[0]]

for _ in range(20):
    routes=[]

    for _ in range(n):
        r=[0]

        while len(r)<n:
            i=r[-1]
            prob=[p[i][j]/(d[i][j]+1) if j not in r else 0 for j in range(n)]
            prob=np.array(prob)/sum(prob)
            r.append(np.random.choice(range(n),p=prob))

        routes.append(r)

    p*=0.5

    for r in routes:
        for i in range(n-1):
            p[r[i]][r[i+1]]+=1/f(r)

best=min(routes,key=f)

print("Shortest Route:",best+[best[0]])
print("Minimum Distance:",f(best))


# DC 1 : Design a distributed application using RPC for remote computation where client submits an integer value to the server and server calculates factorial and returns the result to the client program.

# server.py
from xmlrpc.server import SimpleXMLRPCServer

def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

server = SimpleXMLRPCServer(("Localhost", 1234))
server.register_function(factorial)
server.serve_forever()

# client.py
import xmlrpc.client

proxy = xmlrpc.client.ServerProxy("http://localhost:1234")
print("Factorial is :", proxy.factorial(5))


# DC 3 : Write code to simulate requests coming from clients and distribute them among the servers using the load balancing algorithms

import random

servers = ["Server1", "Server2","Server3","Server4","Server5","Server6"]

requests = ["Client1", "Client2", "Client3"]

print("ROUND ROBIN")
for i in range(len(requests)):
	server = servers[i % len(servers)]
	print(requests[i], "assigned to", server)

print("LEAST CONNECTION LOAD BALANCING")
load = {"Server1" : 0, "Server2" : 0, "Server3" : 0}

for request in requests:
	server = min(load, key = load.get)
	print(request, "assigned to", server)
	load[server] += 1
	print("Current load :", load)
	load[server] -= 1

print("Final load\n", load)

print("RANDOM LOAD BALANCING")

for request in requests:
	server = random.choice(servers)
	print(request, "assigned to", server)


# DC 5 :  Design and develop a distributed application to find the coolest/hottest year from the available weather data. Use weather data from the Internet and process it using MapReduce.
import pandas as pd

# Load Dataset
df = pd.read_csv('weather.csv')
print("Original Data:")
print(df.head())

#  Extract Year from Date (format: DD.MM.YY)
# Date looks like "23.1.01" → split('.') → ['23','1','01'] → index[2] = '01' (year)
df['Year'] = df['Date'].apply(lambda x: x.split('.')[2])

print("\nData with Year column:")
print(df)

# Check for nulls
print("\nNull Values:")
print(df.isnull().sum())

# MAP PHASE
def mapper(df):
    result = []
    for _, row in df.iterrows():
        year = row['Year']
        temp = row['Temperature']
        result.append((year, temp))   # emit (year, temperature) pairs
    return result

mapped_output = mapper(df)
print("\nMapped Output:")
print(mapped_output)

# REDUCE PHASE
def reducer(mapped_output):
    grouped = {}
    for year, temp in mapped_output:
        if year in grouped:
            grouped[year].append(temp)
        else:
            grouped[year] = [temp]
    # compute average temperature per year
    avg_by_year = {year: round(sum(temps) / len(temps), 2)
                   for year, temps in grouped.items()}
    return avg_by_year

reducer_output = reducer(mapped_output)
print("\nReduced Output (Avg Temp per Year):")
for year, avg in sorted(reducer_output.items()):
    print(f"  Year {year}: {avg}°C")

# FIND HOTTEST & COLDEST YEAR
hottest_year = max(reducer_output, key=reducer_output.get)
print(f"Hottest Year: {hottest_year} with avg temp = {reducer_output[hottest_year]}°C")

coldest_year = min(reducer_output, key=reducer_output.get)
print(f"Coldest Year: {coldest_year} with avg temp = {reducer_output[coldest_year]}°C")