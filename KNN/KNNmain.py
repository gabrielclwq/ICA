from cv2 import distanceTransform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

def dEuclidian(p, q):
    p = np.array(p)
    q = np.array(q)
    s = p - q
    s = s**2
    return (s.sum())**(1/2)

def dManhattan(p, q):
    p = np.array(p)
    q = np.array(q)
    s = np.abs(p - q)
    return s.sum()

def dMinkowski(p, q, r):
    p = np.array(p)
    q = np.array(q)
    s = p - q
    s = s**r
    return (s.sum())**(1/r)

def organizador(my_list1, my_list2):
    for i in range(len(my_list1)):
        for j in range(i + 1, len(my_list1)):
            if my_list1[i] > my_list1[j]:
                my_list1[i], my_list1[j] = my_list1[j], my_list1[i]
                my_list2[i], my_list2[j] = my_list2[j], my_list2[i]
    return(my_list1, my_list2)

# Importação do banco de dados a ser utilizado nos exemplos

iris = datasets.load_iris()

# Nesse banco de dados, os principais valores que iremos utilizar serão as entradas (iris.data) e as saídas (iris.target)

X = iris.data

Y = iris.target

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

kn = 3
pred = []
for i in X_test:
    distances = []
    index = []
    for j in range(len(X_train)):
        distances.append(dEuclidian(i, X_train[j]))
        index.append(j)
    distances, index = organizador(distances, index)
    distances = np.array(distances[0:kn])
    index = np.array(index[0:kn])
    print(distances)
    print(index)
    y_train = np.array(Y_train[index])
    print(y_train)
    m = ((distances*y_train).sum())/(distances.sum())
    print(m)
    pred.append(round(m))

print(f'Teste: {Y_test}\nPred:  {np.array(pred)}')