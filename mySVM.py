from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from svm_smo import modelo
import pandas as pd

"""svm.smo.py: Support Vector Machine implementation with smo"""

__author__ = "Gabriel Costa Leite"
__email__ = "gabriel.wq@alu.ufc.br"
__maintainer__ = "Gabriel Costa Leite"
__status__ = "Production"

data = pd.read_csv("car.data")

predict = 'class'

Y = data[predict]
X = data.drop(predict, 1)

#Transformando string em int:
X_set = []
for i in X.columns:
    x_set = list(set(X[i]))
    a = 1
    for j in x_set:
        X[i] = list(map(lambda x: a if x == j else x , X[i]))
        a += 1
    X_set.append(x_set)

Y_set = []
y_set = list(set(Y))
a = 1
for j in y_set:
    Y = list(map(lambda x: a if x == j else x , Y))
    a += 1
Y_set.append(y_set)

#Normalizando X:
for i in X.columns:
    X[i] = X[i]/max(X[i])

""" #Desnormalizar X:
for i in X.columns:
    X[i] = X[i]/min(X[i]) """

#Treino

X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y, test_size=0.3)

y_pred, tempoTreino, tempoTeste, tempoTotal = modelo(X_train, Y_train, X_test, returnTime=1)

print(f'Previsão:\n{y_pred}')
print(f'Teste:\n{Y_test}')
y_pred = np.array(y_pred)
Y_test = np.array(Y_test)

print(f'Tempo de treino: {tempoTreino} | Tempo de teste: {tempoTeste} | Tempo total: {tempoTotal}')

accuracy = metrics.accuracy_score(Y_test, y_pred)

print(f'\tAccuracy: {accuracy}')

plt.figure()
plt.scatter(range(len(Y_test)), Y_test, label="Y_test", color="k")
plt.scatter(range(len(Y_test)), y_pred, label="Y_pred", marker='x', color="r")
plt.legend()
plt.grid()
plt.show()

#Treino sklearn

X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y, test_size=0.3)

clf = SVC(C=1, gamma=1)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
accuracy = metrics.accuracy_score(Y_test, Y_pred)

print(f'Previsão:\n{Y_pred}')
print(f'Teste:\n{Y_test}')

#print(f'Tempo de treino: {tempoTreino} | Tempo de teste: {tempoTeste} | Tempo total: {tempoTotal}')

accuracy = metrics.accuracy_score(Y_test, Y_pred)

print(f'\tAccuracy: {accuracy}')

plt.figure()
plt.scatter(range(len(Y_test)), Y_test, label="Y_test", color="k")
plt.scatter(range(len(Y_test)), Y_pred, label="Y_pred", marker='x', color="r")
plt.legend()
plt.grid()
plt.show()

""" X = iris.data[:, :2]  # we only take the first two features. We could
# avoid this ugly slicing by using a two-dim dataset
Y = iris.target

Y_train = Y
X_train = X

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
X_test = np.c_[xx.ravel(), yy.ravel()]

y_pred = modelo(X_train, Y_train, X_test)

plt.figure()
# Put the result into a color plot
Z = y_pred.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors="k")
plt.title("3-Class classification using Support Vector Machine with custom kernel")
plt.axis("tight")

plt.show() """