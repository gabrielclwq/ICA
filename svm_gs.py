from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import time

"""svm.smo.py: Support Vector Machine implementation with smo"""

__author__ = "Gabriel Costa Leite"
__email__ = "gabriel.wq@alu.ufc.br"
__maintainer__ = "Gabriel Costa Leite"
__status__ = "Production"

class svm_gs():

    def __init__(self, X, Y, C=1, tol=1e-5, max_iter=3, gamma=1): #Inicializa com X e Y de treinamento
        self.X = X
        self.Y = Y
        self.C = C
        self.gamma = gamma
        self.a = np.zeros(len(self.Y))
        self.b = 0
        self.tol = tol
        self.max_iter = max_iter

        iter = 0
        
    def optimizacao_alfa(self):
        firstTarget = np.random.randint(len(self.Y))

        self.a[firstTarget] = self.C

        if self.Y[firstTarget] == 1:
            secondTarget = np.random.randint(len(self.Y))
            while self.Y[secondTarget] != -1:
                secondTarget = np.random.randint(len(self.Y))
            self.a[secondTarget] = self.C
        elif self.Y[firstTarget] == -1:
            secondTarget = np.random.randint(len(self.Y))
            while self.Y[secondTarget] != 1:
                secondTarget = np.random.randint(len(self.Y))
            self.a[secondTarget] = self.C
        
    def kernel(self, x, u):
        return np.exp(-self.gamma*((np.linalg.norm(x-u))**2))

    def decision_func(self, X):
        sum = 0
        for i in range(len(self.Y)):
            sum += self.a[i]*self.Y[i]*self.kernel(self.X[i], X)
        return sum + self.b

    def predict(self, X):
        if self.decision_func(X) >= 0:
            return 1
        else:
            return -1

    def get_alpha(self): return self.a
    def get_b(self): return self.b

def modelo(X_train, Y_train, X_test, returnTime=0):
    t1 = time.time()
    #Pega os valores das classes no dataset em estudo
    y_set = list(set(Y_train))

    #Define os duelos entre as classes sem permetir repetição
    setClasses = []
    for i in range(len(y_set)-1):
        for j in range(i+1,len(y_set)):
            setClasses.append([y_set[i], y_set[j]])

    #Para cada duelo, iremos mapear os valores da classe como -1 e 1

    tempoTreino = 0
    tempoTeste = 0
    res = []
    for s in setClasses:
        t2 = time.time()
        x_train = X_train.copy().tolist()
        y_train = list(map(lambda x: -1 if x == s[0] else (1 if x == s[1] else None) , Y_train))
        for i in range(len(y_train)):
            if y_train[i] == None:
                x_train[i] = None
        y_train = list(filter(lambda x: x != None, y_train))
        x_train = list(filter(lambda i: i != None, x_train))

        t = time.time()
        svm = svm_smo(np.array(x_train), np.array(y_train), tol=1e-3, max_iter=100)
        print(time.time() - t)
        tempoTreino += time.time() - t2

        t3 = time.time()
        Y_pred = []

        for i in range(len(X_test)):
            y_pred = svm.predict(X_test[i])
            Y_pred.append(y_pred)

        Y_pred = list(map(lambda x: s[0] if x == -1 else (s[1] if x == 1 else None) , Y_pred))

        res.append(Y_pred)
        tempoTeste += time.time() - t3

    #Obtendo a previsão para cada duelo, temos que achar a melhor que mais se repete:

    res = np.array(res)
    y_pred = stats.mode(res)[0].ravel()
    
    tempoTotal = time.time() - t1

    if returnTime == 1:
        return y_pred, tempoTreino, tempoTeste, tempoTotal
    else:
        return(y_pred)
