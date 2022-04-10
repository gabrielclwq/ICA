from os import P_NOWAIT
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
from SVMAnalytics import SVMAnalytics


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

#Organiza do pior modelo para o melhor:
for i in range(len(models)):
    for j in range(i + 1, len(models)):
        if models[i].getAccuracy() > models[j].getAccuracy():
            models[i], models[j] = models[j], models[i]

#The best model is:
model = models[-1]
print(model.getAccuracy())

matrizConf = model.getMatrizConfusao()
matrizSup = model.getMatrizSuporte()
matrizSup.to_csv("df")
print(matrizConf)
print(matrizSup)

fig1 = model.getAllDataClassifiersGraph()
fig2 = model.getPredDataClassifiersGraph()

""" for i in testsize:
    model = SVMAnalytics(X, Y, C=5, gamma=5, test_size=i)
    matrizConf = model.getMatrizConfusao()
    matrizSup = model.getMatrizSuporte()
    print(matrizConf)
    print(matrizSup['Accuracy'])
    matrizesConf.append(matrizConf)
    matrizesSup.append(matrizSup)
    models.append(model)

    sensitive.append(matrizSup['TPR'].tolist())
    specificity.append(matrizSup['FPR'].tolist())

Xnames, Ynames = model.getNames()

#plot roc:
def organizador(my_list1, my_list2):
    for i in range(len(my_list1)):
        for j in range(i + 1, len(my_list1)):
            if my_list1[i] > my_list1[j]:
                my_list1[i], my_list1[j] = my_list1[j], my_list1[i]
                my_list2[i], my_list2[j] = my_list2[j], my_list2[i]
    return(my_list1, my_list2)
specificity, sensitive = organizador(specificity, sensitive)

sensitive = np.array(sensitive)
specificity = np.array(specificity)

plt.figure()
for i in range(len(Ynames)):
    specificity[:,i] = (specificity[:,i]-min(specificity[:,i]))/(max(specificity[:,i])-min(specificity[:,i]))
    sensitive[:,i] = (sensitive[:,i]-min(sensitive[:,i]))/(max(sensitive[:,i])-min(sensitive[:,i]))
    plt.plot(specificity[:,i], sensitive[:,i])
plt.legend(Ynames) """

plt.show()


