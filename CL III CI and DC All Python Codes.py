############################################################################################################################################################################################################
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

############################################################################################################################################################################################################
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

############################################################################################################################################################################################################
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

############################################################################################################################################################################################################
# CI 4 : Implement DEAP (Distributed Evolutionary Algorithms) using Python

from deap import base, creator, tools, algorithms
import random

creator.create("Fit", base.Fitness, weights=(1.0,))
creator.create("Ind", list, fitness=creator.Fit)

toolbox = base.Toolbox()

toolbox.register("attr", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Ind, toolbox.attr, 5)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", lambda x: (sum(x),))
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

pop = toolbox.population(n=5)

algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 5)

print("Best:", tools.selBest(pop, 1)[0])

############################################################################################################################################################################################################
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

############################################################################################################################################################################################################

############################################################################################################################################################################################################
 # DC 1 : Design a distributed application using RPC for remote computation where client submits an integer value to the server and server calculates factorial and returns the result to the client program.


#server.py
from xmlrpc.server import SimpleXMLRPCServer

def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

server = SimpleXMLRPCServer(("localhost", 8000))
print("RPC Server running on port 8000...")

server.register_function(factorial, "factorial")
server.serve_forever()

#client.py
import xmlrpc.client

proxy = xmlrpc.client.ServerProxy("http://localhost:8000/")

num = int(input("Enter number: "))
result = proxy.factorial(num)

print("Factorial =", result)

############################################################################################################################################################################################################
# DC 2 : Design a distributed application using RMI for remote computation where client submits two strings to the server and server returns the concatenation of the given strings.

#server.py
from xmlrpc.server import SimpleXMLRPCServer

def concat(a, b):
    return a + b

server = SimpleXMLRPCServer(("localhost", 9000))
server.register_function(concat, "concat")
server.serve_forever()

#client.py
import xmlrpc.client

proxy = xmlrpc.client.ServerProxy("http://localhost:9000/")
print(proxy.concat("Hello ", "World"))

############################################################################################################################################################################################################
# DC 3 :  Write code to simulate requests coming from clients and distribute them among the servers using the load balancing algorithms

servers = ["Server1", "Server2", "Server3"]

requests = ["Client1", "Client2", "Client3",
            "Client4", "Client5", "Client6"]

print("Round Robin Load Balancing\n")

for i in range(len(requests)):
    server = servers[i % len(servers)]

    print(requests[i], "assigned to", server)

############################################################################################################################################################################################################
# DC 4 : JAVA Code Skipped



############################################################################################################################################################################################################
# DC 5 :  Design and develop a distributed application to find the coolest/hottest year from the available weather data. Use weather data from the Internet and process it using MapReduce.

import csv

weather_data={}

for row in csv.DictReader(open("weather.csv")):
    year=row["year"]
    temperature=float(row["temperature"])

    if year not in weather_data:
        weather_data[year]=[]

    weather_data[year].append(temperature)

average_temp={year:sum(temp)/len(temp) for year,temp in weather_data.items()}

print("Hottest Year:",max(average_temp,key=average_temp.get))
print("Coolest Year:",min(average_temp,key=average_temp.get))
