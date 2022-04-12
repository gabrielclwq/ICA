from os import P_NOWAIT
from sklearn import datasets, metrics
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
#from svm_smo import modelo
import pandas as pd
import time
from SVMAnalytics import SVMAnalytics
from sklearn.preprocessing import label_binarize
from itertools import cycle
np.seterr(all="ignore")

data = pd.read_csv("car.data")
print(data.head())

predict = 'class'

Y = data[predict]
X = data.drop(predict, 1)

testsize = np.arange(0.1, 1, 0.1)
C = np.arange(1, 10, 1)
gamma = np.arange(1, 10, 1)

models = []
matrizesConf = []
matrizesSup = []
sensitive = []
specificity = []


#Find the best C and the best gamma:
for c in C:
    for g in gamma:
        model = SVMAnalytics(X, Y, C=c, gamma=g)
        models.append(model)
        
        dfC = model.getMatrizConfusao()
        dfS = model.getMatrizSuporte()
        sensitive.append(dfS['TPR'].tolist())
        specificity.append(dfS['FPR'].tolist())

sensitive = np.array(sensitive)
specificity = np.array(specificity)

Xnames, Ynames = model.getNames()

plt.figure()
for i in range(len(Ynames)):
    plt.plot(specificity[:,i], sensitive[:,i])
plt.legend(Ynames)

#Organiza do pior modelo para o melhor:
for i in range(len(models)):
    for j in range(i + 1, len(models)):
        if models[i].getAccuracy() > models[j].getAccuracy():
            models[i], models[j] = models[j], models[i]

#The best model is:
model = models[-1]
print(model.getAccuracy())

C, gamma = model.getConstants()

#Validação do modelo:

vmodels = []
sensitive = []
specificity = []
for i in testsize:
    vmodel = SVMAnalytics(X, Y, C=C, gamma=gamma, test_size=1-i)
    vmodels.append(vmodel)
    dfC = vmodel.getMatrizConfusao()
    dfS = vmodel.getMatrizSuporte()
    sensitive.append(dfS['TPR'].tolist())
    specificity.append(dfS['FPR'].tolist())

    """ print(f'Modelo testsize = {1-i}')
    print(f'Matriz de confusão:\n{dfC}')
    print(f'Matriz de suporte:\n{dfS[["Accuracy", "TPR", "FPR", "F1 score"]]}') """

sensitive = np.array(sensitive)
specificity = np.array(specificity)

Xnames, Ynames = vmodel.getNames()

plt.figure()
for i in range(len(Ynames)):
    """ specificity[:,i] = (specificity[:,i]-min(specificity[:,i]))/(max(specificity[:,i])-min(specificity[:,i]))
    sensitive[:,i] = (sensitive[:,i]-min(sensitive[:,i]))/(max(sensitive[:,i])-min(sensitive[:,i])) """
    plt.plot(specificity[:,i], sensitive[:,i])
plt.legend(Ynames)

#Organiza do pior modelo para o melhor:
for i in range(len(vmodels)):
    for j in range(i + 1, len(vmodels)):
        if vmodels[i].getAccuracy() > vmodels[j].getAccuracy():
            vmodels[i], vmodels[j] = vmodels[j], vmodels[i]

#The best model is:
vmodel = vmodels[-1]
dfC = vmodel.getMatrizConfusao()
dfS = vmodel.getMatrizSuporte()
print(f'Modelo testsize = {vmodel.getTestsize()}')
print(f'Constantes: C = {C}, gamma = {gamma}')
print(f'Tempo de treino: {vmodel.getTempotreino()}')
print(f'Matriz de confusão:\n{dfC}')
print(f'Matriz de suporte:\n{dfS[["Accuracy", "TPR", "FPR", "F1 score"]]}')
test_size = vmodel.getTestsize()

fig1 = vmodel.getAllDataClassifiersGraph()
fig2 = vmodel.getPredDataClassifiersGraph()

data_set = list(set(data[predict].tolist()))

datas = []
for i in data_set:
    a = data.loc[data[predict] == i]
    b = data.loc[data[predict] != i]
    datas.append([a,b])

dfs = []
for i in datas:
    for t in range(2,len(i[0]), 10):
        df = i[0].sample(n=t)
        df = df.append(i[1])
        dfs.append(df)

print(dfs[0])

dmodels = []
dsensitive = []
dspecificity = []

for d in dfs:

    Y = d[predict]
    X = d.drop(predict, 1)

    dmodel = SVMAnalytics(X, Y, C=C, gamma=gamma, test_size=0.2)
    dmodels.append(dmodel)
    ddfC = dmodel.getMatrizConfusao()
    ddfS = dmodel.getMatrizSuporte()
    dsensitive.append(ddfS['TPR'].tolist())
    dspecificity.append(ddfS['FPR'].tolist())

    """ print(f'Matriz de confusão:\n{ddfC}')
    print(f'Matriz de suporte:\n{ddfS[["Accuracy", "TPR", "FPR", "F1 score"]]}') """

dsensitive = np.array(dsensitive)
dspecificity = np.array(dspecificity)

Xnames, Ynames = dmodel.getNames()

plt.figure()
for i in range(len(Ynames)):
    plt.plot(dspecificity[:,i], dsensitive[:,i])
plt.legend(Ynames)

plt.show()

