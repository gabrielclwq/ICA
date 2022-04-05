from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

"""svm.smo.py: Support Vector Machine implementation with smo"""

__author__ = "Gabriel Costa Leite"
__email__ = "gabriel.wq@alu.ufc.br"
__maintainer__ = "Gabriel Costa Leite"
__status__ = "Production"

class svm_smo():

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
        self.smo(iter)
        
    def smo(self, iter):
        while iter < self.max_iter:
            num_changed_alfas = 0

            for i in range(len(self.Y)):
                Ei = self.decision_func(self.X[i]) - self.Y[i]
                
                if (self.Y[i]*Ei < -self.tol and self.a[i] < self.C) or (self.Y[i]*Ei > self.tol and self.a[i] > 0):
                    j = i
                    while(j == i):
                        j = np.random.randint(0, len(self.Y))

                    Ej = self.decision_func(self.X[j]) - self.Y[j]
                    
                    ai_old = self.a[i]
                    aj_old = self.a[j]
                    
                    if(self.Y[i] == self.Y[j]):
                        L = max(0, self.a[j] + self.a[i] - self.C)
                        H = min(self.C, self.a[i] + self.a[j])
                    else:
                        L = max(0, self.a[j] - self.a[i])
                        H = min(self.C , self.C + self.a[j] - self.a[i])
                    
                    if(L == H):
                        continue

                    eta = 2.0*(np.dot(self.X[i],self.X[j])) - np.dot(self.X[i], self.X[i]) - np.dot(self.X[j], self.X[j])

                    if eta >= 0:
                        continue

                    self.a[j] = self.a[j] - (self.Y[j]*(Ei - Ej))/eta

                    if self.a[j] > H:
                        self.a[j] = H
                    elif self.a[j] < L:
                        self.a[j] = L
                    else:
                        self.a[j] = self.a[j]

                    if abs(self.a[j] - aj_old) < 1e-5:
                        continue
                
                    self.a[i] = self.a[i] + self.Y[i] * self.Y[j] * (aj_old - self.a[j])

                    b1 = self.b - Ei - self.Y[i]*(self.a[i] - ai_old)*np.dot(self.X[i], self.X[i]) - self.Y[j]*(self.a[j] - aj_old)*np.dot(self.X[i], self.X[j])
            
                    b2 = self.b - Ej - self.Y[i]*(self.a[i] - ai_old)*np.dot(self.X[i], self.X[j]) - self.Y[j]*(self.a[j] - aj_old)*np.dot(self.X[j], self.X[j])
                    
                    if self.a[i] > 0 and self.a[i] < self.C:
                        self.b = b1
                    elif self.a[j] > 0 and self.a[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2)/2

                    num_changed_alfas += 1
            
            if (num_changed_alfas == 0):
                iter += 1
            else:
                iter = 0

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
