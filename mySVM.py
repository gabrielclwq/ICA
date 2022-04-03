from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

#Definição de funções necessárias

# Definir a função do kernel -> RBF
def K(x, u, gamma):
    k = np.exp(-gamma*((np.linalg.norm(x-u))**2))
    return k

#Define a função de descisão do modelo
def decision(a, y, x, u, sigma, b):
    sum = 0
    for i in range(len(a)):
        sum = sum + a[i]*y[i]*K(x[i], u, sigma)
    f = sum + b

    if f >= 0:
        return 1
    else:
        return -1

#Define a optimização para encontrar os suporte vector
def optimizacao_alfa(C):
    a = np.zeros((1, len(y_train)))
    a = a.ravel()
    firstTarget = np.random.randint(len(y_train))

    a[firstTarget] = C

    if y_train[firstTarget] == 1:
        secondTarget = np.random.randint(len(y_train))
        while y_train[secondTarget] != -1:
            secondTarget = np.random.randint(len(y_train))
        a[secondTarget] = C
    elif y_train[firstTarget] == -1:
        secondTarget = np.random.randint(len(y_train))
        while y_train[secondTarget] != 1:
            secondTarget = np.random.randint(len(y_train))
        a[secondTarget] = C
    return a

# import some data to play with
iris = datasets.load_iris()
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

    #Com isso podemos rodar encontrar a classificação para cada classe:

    Y_pred = []
    a = optimizacao_alfa(10)

    for i in range(len(X_test)):
        y_pred = decision(a, y_train, x_train, X_test[i], 1, 1)
        Y_pred.append(y_pred)

    print(f'Classe = {str(s)}:')

    Y_pred = list(map(lambda x: s[0] if x == -1 else (s[1] if x == 1 else None) , Y_pred))

    print(f'\tPrevisão: {Y_pred}')

    res.append(Y_pred)

#Obtendo a previsão para cada duelo, temos que achar a melhor que mais se repete:

res = np.array(res)
y_pred = stats.mode(res)[0].ravel()

# Put the result into a color plot
Z = y_pred.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors="k")
plt.title("3-Class classification using Support Vector Machine with custom kernel")
plt.axis("tight")
plt.show()

# we create an instance of SVM and fit out data.
clf = svm.SVC(kernel="rbf", gamma=1, C=1)
clf.fit(X, Y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors="k")
plt.title("3-Class classification using Support Vector Machine with custom kernel")
plt.axis("tight")
plt.show()