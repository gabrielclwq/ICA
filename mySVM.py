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
modelos = []

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

""" #Desnormalizar X:
for i in X.columns:
    X[i] = X[i]/min(X[i]) """

""" #Treino

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
plt.show() """

#Treino sklearn

#Divide o dataset em treino/treino e validacao
X_tt, X_val, Y_tt, Y_val = train_test_split(X.values, Y, test_size=0.3)

#Define o intervalo de C e gamma para ser testado
C = range(1,10)
gamma = range(1,10)
test_size = np.arange(0.1, 0.9, 0.01)
teste = []

#Divide o conjunto de treino em um treino e teste
for ts in test_size:
    modelos = []

    X_train, X_test, Y_train, Y_test = train_test_split(X_tt, Y_tt, test_size=ts)

    tempoTreino = time.time()
    for c in C:
        for g in gamma:
            tempoModelo = time.time()
            clf = SVC(C=c, gamma=g)
            clf.fit(X_train, Y_train)
            y_pred = clf.predict(X_test)
            acertos = 0
            for i in range(len(y_pred)):
                if y_pred[i] == Y_test[i]:
                    acertos += 1
                else:
                    pass
            acc = acertos/(len(y_pred))
            tempoModelo = time.time() - tempoModelo
            modelos.append([acc, clf, c, g, acertos, tempoModelo, [X_train, X_test, Y_train, Y_test]])
    tempoTreino = time.time() - tempoTreino

    # Agora, obtidos os modelos, iremos ordenar de forma crescente tendo em vista a accuracy:
    for i in range(len(modelos)):
        for j in range(i + 1, len(modelos)):
            if modelos[i][0] > modelos[j][0]:
                modelos[i], modelos[j] = modelos[j], modelos[i]

    #O melhor modelo será utilizado para a validação:
    modelo = modelos[-1]
    clf = modelo[1]
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

    teste.append([modelo, ts, tempoTreino, y_val, dfMatrizConf, dfMatrizSuporte])

#print:
sensitive = []
specificity = []
for i in teste:
    modelo = i[0]
    dfMatrizConf = i[-2]
    dfMatrizSuporte = i[-1]
    print(f'Modelo: {modelo[1]} | Tempo de treinamento: {i[2]}')
    print(f'Matriz de confusão | Size of Train/Test: {len(X_tt)} -> Train % = {1-i[1]} - Test % = {i[1]} | Size of Validation: {len(X_val)}:')
    print(dfMatrizConf)
    print(f'Estatísticas | Size of Train/Test: {len(X_tt)} -> Train % = {1-i[1]} - Test % = {i[1]} | Size of Validation: {len(X_val)}:')
    print(dfMatrizSuporte)
    print("\n")

    sensitive.append(dfMatrizSuporte['recall'].tolist())
    specificity.append(dfMatrizSuporte['specificity'].tolist())



#data_plot:
fig, axs = plt.subplots(2)
fig.suptitle('Data Classifiers')

def organizador(my_list1, my_list2):
    for i in range(len(my_list1)):
        for j in range(i + 1, len(my_list1)):
            if my_list1[i] > my_list1[j]:
                my_list1[i], my_list1[j] = my_list1[j], my_list1[i]
                my_list2[i], my_list2[j] = my_list2[j], my_list2[i]
    return(my_list1, my_list2)

Y_, X_ = organizador(Y, X.values)
lens = [0]
setY_ = list(set(Y_))
soma = 0
for i in range(len(setY_)):
    for j in range(len(Y_)):
        if Y_[j] == setY_[i]:
            soma += 1
    lens.append(soma)

for i in range(len(setY_)):
    axs[0].scatter(X_[lens[i]:lens[i+1]-1, 0], X_.sum(axis=1)[lens[i]:lens[i+1]-1])
    axs[1].scatter(X_[lens[i]:lens[i+1]-1, 0], X_.sum(axis=1)[lens[i]:lens[i+1]-1]+i*10)
axs[0].legend(name)
axs[0].set_title("Real")
axs[1].set_title("Distribuido")
for ax in axs.flat:
    ax.set(xlabel='soma de atribudos', ylabel='buying')
for ax in axs.flat:
    ax.label_outer()

#plot roc:
sensitive = np.array(sensitive)
specificity = np.array(specificity)

plt.figure()
for i in range(len(name)):
    specificity[:,i] = (specificity[:,i]-min(specificity[:,i]))/(max(specificity[:,i])-min(specificity[:,i]))
    sensitive[:,i] = (sensitive[:,i]-min(sensitive[:,i]))/(max(sensitive[:,i])-min(sensitive[:,i]))
    plt.plot(specificity[:,i], sensitive[:,i])
plt.legend(name)

plt.show()






""" print(confusion_matrix(Y_val, y_val))
print(classification_report(Y_val, y_val))
 """
""" plt.figure()
plt.scatter(range(len(Y_test)), Y_test, label="Y_test", color="k")
plt.scatter(range(len(Y_test)), Y_pred, label="Y_pred", marker='x', color="r")
plt.legend()
plt.grid()
plt.show() """

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