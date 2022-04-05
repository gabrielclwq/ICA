from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from svm_smo import svm_smo as SVM

"""svm.smo.py: Support Vector Machine implementation with smo"""

__author__ = "Gabriel Costa Leite"
__email__ = "gabriel.wq@alu.ufc.br"
__maintainer__ = "Gabriel Costa Leite"
__status__ = "Production"

iris = datasets.load_iris()

X_iris = iris.data

y_iris = iris.target

#One-vs-one classifier method

X_train, X_test, Y_train, Y_test = train_test_split(X_iris, y_iris, test_size=0.3)

#Pega os valores das classes no dataset em estudo
y_set = list(set(Y_train))

#Define os duelos entre as classes sem permetir repetição
setClasses = []
for i in range(len(y_set)-1):
    for j in range(i+1,len(y_set)):
        setClasses.append([y_set[i], y_set[j]])

#Para cada duelo, iremos mapear os valores da classe como -1 e 1

res = []
for s in setClasses:
    x_train = X_train.copy().tolist()
    y_train = list(map(lambda x: -1 if x == s[0] else (1 if x == s[1] else None) , Y_train))
    for i in range(len(y_train)):
        if y_train[i] == None:
            x_train[i] = None
    y_train = list(filter(lambda x: x != None, y_train))
    x_train = list(filter(lambda i: i != None, x_train))

    svm = SVM(np.array(x_train), np.array(y_train))

    Y_pred = []

    for i in range(len(X_test)):
        y_pred = svm.predict(X_test[i])
        Y_pred.append(y_pred)

    print(f'Classe = {str(s)}:')

    Y_pred = list(map(lambda x: s[0] if x == -1 else (s[1] if x == 1 else None) , Y_pred))

    print(f'\tPrevisão: {Y_pred}')

    res.append(Y_pred)

#Obtendo a previsão para cada duelo, temos que achar a melhor que mais se repete:

res = np.array(res)
y_pred = stats.mode(res)[0].ravel()

print(f'Previsão: {y_pred}')
print(f'Teste: {Y_test}')
y_pred = np.array(y_pred)
Y_test = np.array(Y_test)

accuracy = metrics.accuracy_score(Y_test, y_pred)

print(f'\tAccuracy: {accuracy}')

plt.figure()
plt.scatter(range(len(Y_test)), Y_test, label="Y_test", color="k")
plt.scatter(range(len(Y_test)), y_pred, label="Y_pred", marker='x', color="r")
plt.legend()
plt.grid()

clf = SVC(kernel="rbf", C=1, gamma=1)
clf.fit(X_train, Y_train)

accuracy = metrics.accuracy_score(Y_test, clf.predict(X_test))

print(f'\tAccuracy sklearn: {accuracy}')

plt.figure()
plt.scatter(range(len(Y_test)), Y_test, label="Y_test", color="k")
plt.scatter(range(len(Y_test)), clf.predict(X_test), label="Y_pred", marker='x', color="r")
plt.legend()
plt.grid()
plt.show()

X = iris.data[:, :2]  # we only take the first two features. We could
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

#Pega os valores das classes no dataset em estudo
y_set = list(set(Y_train))

#Define os duelos entre as classes sem permetir repetição
setClasses = []
for i in range(len(y_set)-1):
    for j in range(i+1,len(y_set)):
        setClasses.append([y_set[i], y_set[j]])

#Para cada duelo, iremos mapear os valores da classe como -1 e 1

res = []
for s in setClasses:
    x_train = X_train.copy().tolist()
    y_train = list(map(lambda x: -1 if x == s[0] else (1 if x == s[1] else None) , Y_train))
    for i in range(len(y_train)):
        if y_train[i] == None:
            x_train[i] = None
    y_train = list(filter(lambda x: x != None, y_train))
    x_train = list(filter(lambda i: i != None, x_train))

    svm = SVM(np.array(x_train), np.array(y_train))

    Y_pred = []

    for i in range(len(X_test)):
        y_pred = svm.predict(X_test[i])
        Y_pred.append(y_pred)

    print(f'Classe = {str(s)}:')

    Y_pred = list(map(lambda x: s[0] if x == -1 else (s[1] if x == 1 else None) , Y_pred))

    print(f'\tPrevisão: {Y_pred}')

    res.append(Y_pred)

#Obtendo a previsão para cada duelo, temos que achar a melhor que mais se repete:

res = np.array(res)
y_pred = stats.mode(res)[0].ravel()

plt.figure()
# Put the result into a color plot
Z = y_pred.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors="k")
plt.title("3-Class classification using Support Vector Machine with custom kernel")
plt.axis("tight")

# we create an instance of SVM and fit out data.
clf = SVC(kernel="rbf", gamma=1, C=1)
clf.fit(X, Y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
plt.figure()
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors="k")
plt.title("3-Class classification using Support Vector Machine with custom kernel")
plt.axis("tight")
plt.show()