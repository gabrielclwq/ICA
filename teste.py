from sklearn import datasets, metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
#from svm_smo import modelo
import pandas as pd
import time

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

name = Y_set[0]

#Treino sklearn

#Divide o dataset em treino/treino e validacao
X_tt, X_val, Y_tt, Y_val = train_test_split(X.values, Y, test_size=0.3)

#Define o intervalo de C e gamma para ser testado
C = [5]
gamma = [5]
test_size = np.arange(0.1, 1, 0.1)
modelos = []
teste = []

#Divide o conjunto de treino em um treino e teste
for ts in test_size:
    X_train, X_test, Y_train, Y_test = train_test_split(X_tt, Y_tt, test_size=ts)

    clf = SVC(C=5, gamma=5)
    clf.fit(X_train, Y_train)
    y_pred = clf.predict(X_test)
    acertos = 0
    for i in range(len(y_pred)):
        if y_pred[i] == Y_test[i]:
            acertos += 1
        else:
            pass
    acc = acertos/(len(y_pred))

    y_val = clf.predict(X_val)

    #Obtendo a matriz de confusão:
    y_set = list(set(Y_val))
    matrizConf = np.zeros((len(y_set), len(y_set)))
    for i in range(len(y_set)): #i -> real
        for j in range(len(y_set)): #j -> previsao
            nReal = 0 #total
            nPrev = 0 #acertos
            for k in range(len(Y_val)):
                if Y_val[k] == y_set[i]:
                    nReal += 1
                    if y_val[k] == y_set[j]:
                        nPrev += 1
            matrizConf[i,j] = nPrev

    #Renomeando a saida:

    dfMatrizConf = pd.DataFrame(matrizConf, columns=name)
    dfMatrizConf.index = name
    #print(dfMatrizConf)

    #Accuracy:
    acc = (matrizConf.diagonal().sum())/(matrizConf.sum())
    #print(f'acc = {acc}')

    #Recall or sensitive:
    recall = []
    for i in range(len(matrizConf[0])):
        recall.append((matrizConf[i,i])/(matrizConf[i].sum()))
    #print(f'recall = {recall}')

    #Precision:
    precision = []
    for i in range(len(matrizConf[0])):
        precision.append((matrizConf[i,i])/(matrizConf[:,i].sum()))
    #print(f'precision = {precision}')

    #fscore:
    fscore = []
    for i in range(len(matrizConf[0])):
        r = (matrizConf[i,i])/(matrizConf[i].sum())
        p = (matrizConf[i,i])/(matrizConf[i].sum())
        fscore.append(2*r*p/(r+p))
    #print(f'fscore = {fscore}')

    #Specificity:
    spec = []
    for i in range(len(matrizConf[0])):
        spec.append((matrizConf.sum() - matrizConf[i].sum() - matrizConf[:,i].sum() + matrizConf[i,i])/(matrizConf.sum() - matrizConf[:,i].sum()))
    #print(f'specificity = {spec}')

    #True positive:
    truePositive = []
    for i in range(len(matrizConf[0])):
        truePositive.append(matrizConf[i,i])
    #print(f'truePositive = {truePositive}')

    #True negative:
    trueNegative = []
    for i in range(len(matrizConf[0])):
        trueNegative.append(matrizConf.sum() - matrizConf[i].sum() - matrizConf[:,i].sum() + matrizConf[i,i])
    #print(f'trueNegative = {trueNegative}')

    matrizSuporte = np.array([precision, recall, fscore, spec, truePositive, trueNegative])
    dfMatrizSuporte = pd.DataFrame(matrizSuporte.T, columns=["precision", "recall", "fscore", "specificity", "truePositive", "trueNegative"])
    dfMatrizSuporte.index = name
    #print(dfMatrizSuporte)

    print(f'dfMatrizConf:\n {dfMatrizConf}')
    print(confusion_matrix(Y_val, y_val))
    print(f'dfMatrizSuporte:\n {dfMatrizSuporte}')

