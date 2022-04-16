from pprint import pformat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
import time
from KNNAnalytics import KNN
np.seterr(all="ignore")

#%% Load data and preprocessing

data = pd.read_csv("bodyfat.csv")
print(data.head())

# Nesse banco de dados, os principais valores que iremos utilizar serão as entradas (iris.data) e as saídas (iris.target)

predict = 'BodyFat'

Y = data[predict]
X = data.drop(columns=[predict])
Xlabels = X.columns

Y = Y.transform(lambda x: 1 if x <= 5 else (2 if x>5 and x<=14 else (3 if x>14 and x<16 else(4 if x>=16 and x<25 else 5))))

Y = Y.to_numpy()
X = X.to_numpy()

#%% Find the best split train/test for the data

accTestsize = []
modelosTestsize = []

for i in range(10):
    for t in np.arange(0.1, 1, 0.1):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=t)
        clf = KNN()

        tempoTreino = time.time()
        clf.fit(X_train, Y_train, Xlabels)
        tempoTreino = time.time() - tempoTreino

        tempoTeste = time.time()
        pred, p = clf.predict(X_test)
        tempoTeste = time.time() - tempoTeste

        acc = clf.getAccuracy(pred, Y_test)

        modelosTestsize.append([clf, t, acc, tempoTreino, tempoTeste])
        accTestsize.append(acc)

for i in range(len(accTestsize)):
    for j in range(i + 1, len(accTestsize)):
        if accTestsize[i] > accTestsize[j]:
            modelosTestsize[i], modelosTestsize[j] = modelosTestsize[j], modelosTestsize[i]
            accTestsize[i], accTestsize[j] = accTestsize[j], accTestsize[i]

bestTestsize = modelosTestsize[-1]
testsize = bestTestsize[1]

modelosTestsize = np.array(modelosTestsize)
fig, axs = plt.subplots(3)
fig.suptitle('Statistics of Grid Search in TestSize')

axs[0].plot(modelosTestsize[:, 2], modelosTestsize[:, 1], label="TestSize")
axs[0].set_ylabel("TestSize")
axs[0].legend()
axs[0].grid()

axs[1].plot(modelosTestsize[:, 2], modelosTestsize[:, 3], label="Tempo de treino")
axs[1].set_ylabel("Tempo de treino")
axs[1].legend()
axs[1].grid()

axs[2].plot(modelosTestsize[:, 2], modelosTestsize[:, 4], label="Tempo de teste")
axs[2].set_ylabel("Tempo de teste")
axs[2].legend()
axs[2].grid()
axs[2].set_xlabel("Accuracy")
for ax in axs.flat:
    ax.label_outer()

#%% Find the best paramns kn and r for the data

accParamns = []
modelosParamns = []

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=testsize)

for r in range(1, 3):
    for k in range(1, 21):
        clf = KNN(kn=k, r=r)

        tempoTreino = time.time()
        clf.fit(X_train, Y_train, Xlabels)
        tempoTreino = time.time() - tempoTreino

        tempoTeste = time.time()
        pred, p = clf.predict(X_test)
        tempoTeste = time.time() - tempoTeste

        acc = clf.getAccuracy(pred, Y_test)

        modelosParamns.append([clf, k, r, acc, tempoTreino, tempoTeste])
        accParamns.append(acc)

for i in range(len(accParamns)):
    for j in range(i + 1, len(accParamns)):
        if accParamns[i] > accParamns[j]:
            modelosParamns[i], modelosParamns[j] = modelosParamns[j], modelosParamns[i]
            accParamns[i], accParamns[j] = accParamns[j], accParamns[i]

bestParamns = modelosParamns[-1]
kn = bestParamns[1]
r = bestParamns[2]

modelosParamns = np.array(modelosParamns)
fig, axs = plt.subplots(4)
fig.suptitle('Statistics of Grid Search in TestSize')

axs[0].plot(modelosParamns[:, 3], modelosParamns[:, 1], label="kn")
axs[0].set_ylabel("kn")
axs[0].legend()
axs[0].grid()

axs[1].plot(modelosParamns[:, 3], modelosParamns[:, 2], label="r")
axs[1].set_ylabel("r")
axs[1].legend()
axs[1].grid()

axs[2].plot(modelosParamns[:, 3], modelosParamns[:, 4], label="Tempo de treino")
axs[2].set_ylabel("Tempo de treino")
axs[2].legend()
axs[2].grid()

axs[3].plot(modelosParamns[:, 3], modelosParamns[:, 5], label="Tempo de teste")
axs[3].set_ylabel("Tempo de teste")
axs[3].legend()
axs[3].grid()
axs[3].set_xlabel("Accuracy")
for ax in axs.flat:
    ax.label_outer()

#%% Get all analysis from the best model find

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=testsize)

clf = KNN(kn=kn, r=r)

tempoTreino = time.time()
clf.fit(X_train, Y_train, Xlabels)
tempoTreino = time.time() - tempoTreino

tempoTeste = time.time()
pred, p = clf.predict(X_test)
tempoTeste = time.time() - tempoTeste

matrizConf = clf.matrizConfusao(pred, Y_test)
matrizSup, supportNames  = clf.matrizSuporte(matrizConf)

dfMatrizConf = pd.DataFrame(matrizConf, columns=clf.getTargetLabel())
dfMatrizConf.index = clf.getTargetLabel()

dfMatrizSup = pd.DataFrame(matrizSup.T, columns=supportNames)
dfMatrizSup.index = clf.getTargetLabel()

print("---------------------------------")
print(f'Parâmetros: kn = {kn}, r = {r}, tempo de teste = {tempoTeste} e tempo de treino = {tempoTreino}')
print("---------------------------------")

print("---------------------------------")
print(f'Y_test =   {Y_test}\nPredição = {pred}')
print("---------------------------------")

print("---------------------------------")
print("Matriz Confusão:")
print(dfMatrizConf)
print("---------------------------------")

print("---------------------------------")
print("Matriz Suporte:")
print(dfMatrizSup[["Accuracy", "TPR", "FPR", "F1 score"]])
print("---------------------------------")

#%% Distrubuition of data

Xd = X.copy()
Yd = Y.copy()
Xd_test = X_test.copy()
predD = pred.copy()

TargetLabels = clf.getTargetLabel()

for i in range(len(Yd)):
    for j in range(i + 1, len(Yd)):
        if Yd[i] > Yd[j]:
            Yd[i], Yd[j] = Yd[j], Yd[i]
            Xd[i], Xd[j] = Xd[j], Xd[i]
for i in range(len(predD)):
    for j in range(i + 1, len(predD)):
        if predD[i] > predD[j]:
            predD[i], predD[j] = predD[j], predD[i]
            Xd_test[i], Xd_test[j] = Xd_test[j], Xd_test[i]

lens = [0]
setY_ = list(set(Yd))
soma = 0
for i in range(len(setY_)):
    for j in range(len(Yd)):
        if Yd[j] == setY_[i]:
            soma += 1
    lens.append(soma)

plt.figure()
for i in range(len(setY_)):
    plt.scatter(Xd[lens[i]:lens[i+1]-1, 0], Xd[lens[i]:lens[i+1]-1, 2])
plt.legend(TargetLabels)
plt.ylabel(Xlabels[2])
plt.title("All Data Distribution")
plt.xlabel(Xlabels[0])
plt.grid()

#%% Feature importance with Permutation

modelosSuffle = []
mseSuffle = []

for i in range(len(Xlabels)):
    Xs_train = X_train.copy()

    #Suflling the collumn:
    np.random.shuffle(Xs_train[:, i].T)

    clfs = KNN(kn=kn, r=r)

    tempoTreino = time.time()
    clfs.fit(Xs_train, Y_train, Xlabels)
    tempoTreino = time.time() - tempoTreino

    tempoTeste = time.time()
    preds, ps = clfs.predict(X_test)
    tempoTeste = time.time() - tempoTeste

    mse = clfs.MSE(preds, Y_test)

    modelosSuffle.append([clfs, i, mse, tempoTeste, tempoTreino])
    mseSuffle.append(mse)

for i in range(len(mseSuffle)):
    for j in range(i + 1, len(mseSuffle)):
        if mseSuffle[i] > mseSuffle[j]:
            modelosSuffle[i], modelosSuffle[j] = modelosSuffle[j], modelosSuffle[i]
            mseSuffle[i], mseSuffle[j] = mseSuffle[j], mseSuffle[i]

modelosSuffle = np.array(modelosSuffle)
fig, axs = plt.subplots(3)
fig.suptitle('Feature importance with Permutation')

axs[0].plot(modelosSuffle[:, 2], modelosSuffle[:, 1], label="Feature")
axs[0].set_ylabel("Feature")
axs[0].legend()
axs[0].grid()

axs[1].plot(modelosParamns[:, 2], modelosParamns[:, 3], label="Tempo de teste")
axs[1].set_ylabel("Tempo de teste")
axs[1].legend()
axs[1].grid()

axs[2].plot(modelosParamns[:, 2], modelosParamns[:, 4], label="Tempo de treino")
axs[2].set_ylabel("Tempo de treino")
axs[2].legend()
axs[2].grid()
axs[2].set_xlabel("MSE")
for ax in axs.flat:
    ax.label_outer()

#%% Get ROC curve analyses:

plt.figure()
plt.plot([0, 1], [0, 1], linestyle='dashed', color="k")

for i in clf.getTargetLabel():
    fpr, tpr, A = clf.ROC(X_test, Y_test, i)
    plt.plot(fpr, tpr, label=f'ROC Curve Class {i} (area = {A})')
    plt.legend()
    plt.grid()

plt.ylabel("True Positeve Rate")
plt.xlabel("False Positive Rate")
plt.title(f'Receiver Operating Characteristic')
plt.show()