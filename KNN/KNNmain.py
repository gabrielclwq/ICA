from pprint import pformat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
np.seterr(all="ignore")

def dEuclidian(p, q):
    p = np.array(p)
    q = np.array(q)
    s = p - q
    s = s**2
    return (s.sum())**(1/2)

def dManhattan(p, q):
    p = np.array(p)
    q = np.array(q)
    s = np.abs(p - q)
    return s.sum()

def dMinkowski(p, q, r):
    p = np.array(p)
    q = np.array(q)
    s = p - q
    s = s**r
    return (s.sum())**(1/r)

def organizador(my_list1, my_list2):
    for i in range(len(my_list1)):
        for j in range(i + 1, len(my_list1)):
            if my_list1[i] > my_list1[j]:
                my_list1[i], my_list1[j] = my_list1[j], my_list1[i]
                my_list2[i], my_list2[j] = my_list2[j], my_list2[i]
    return(my_list1, my_list2)

def knn(X_train, Y_train, X_test, kn=3, r=2):
    pred = []
    p = []
    for i in X_test:
        distances = []
        index = []
        for j in range(len(X_train)):
            distances.append(dMinkowski(i, X_train[j], r))
            index.append(j)
        distances, index = organizador(distances, index)
        distances = np.array(distances[0:kn])
        index = np.array(index[0:kn])
        y_train = np.array(Y_train[index])
        m = ((distances*y_train).sum())/(distances.sum())
        p.append(m)
        pred.append(round(m))
    return pred, p

def matrizConfusao(pred, Y_test):
    y_set = list(set(Y_test))
    matrizConf = np.zeros((len(y_set), len(y_set)))
    for i in range(len(y_set)): #i -> real
        for j in range(len(y_set)): #j -> previsao
            nReal = 0 #total
            nPrev = 0 #acertos
            for k in range(len(Y_test)):
                if Y_test[k] == y_set[i]:
                    nReal += 1
                    if pred[k] == y_set[j]:
                        nPrev += 1
            matrizConf[i,j] = nPrev
    return matrizConf

def matrizSuporte(matrizConf):
    supportName = []
    matrizSup = []

    #Positive Prediction:
    PP = []
    for i in range(len(matrizConf[0])):
        PP.append(matrizConf[:,i].sum())
    supportName.append("PP")
    PP = np.array(PP)
    matrizSup.append(PP)

    #Negative Prediction:
    PN = []
    for i in range(len(matrizConf[0])):
        PN.append(matrizConf.sum() - matrizConf[:,i].sum())
    supportName.append("PN")
    PN = np.array(PN)
    matrizSup.append(PN)

    #Actual Positive:
    AP = []
    for i in range(len(matrizConf[0])):
        AP.append(matrizConf[i].sum())
    supportName.append("AP")
    AP = np.array(AP)
    matrizSup.append(AP)

    #Actual Negative:
    AN = []
    for i in range(len(matrizConf[0])):
        AN.append(matrizConf.sum() - matrizConf[i].sum())
    supportName.append("AN")
    AN = np.array(AN)
    matrizSup.append(AN)

    #True positive:
    TP = []
    for i in range(len(matrizConf[0])):
        TP.append(matrizConf[i,i])
    supportName.append("TP")
    TP = np.array(TP)
    matrizSup.append(TP)

    #False positive:
    FP = []
    for i in range(len(matrizConf[0])):
        FP.append(matrizConf[:,i].sum() - matrizConf[i,i])
    supportName.append("FP")
    FP = np.array(FP)
    matrizSup.append(FP)

    #False positive:
    FN = []
    for i in range(len(matrizConf[0])):
        FN.append(matrizConf[i].sum() - matrizConf[i,i])
    supportName.append("FN")
    FN = np.array(FN)
    matrizSup.append(FN)

    #True negative:
    TN = []
    for i in range(len(matrizConf[0])):
        TN.append(matrizConf.sum() - matrizConf[i].sum() - matrizConf[:,i].sum() + matrizConf[i,i])
    supportName.append("TN")
    TN = np.array(TN)
    matrizSup.append(TN)

    #Prevalence:
    prevalence = AP/(AP+AN)
    supportName.append("Prevalence")
    matrizSup.append(prevalence)

    #Accuracy:
    accuracy = (TP+TN)/(AP+AN)
    supportName.append("Accuracy")
    matrizSup.append(accuracy)

    #Positive predictive value (PPV):
    PPV = TP/PP
    supportName.append("PPV")
    matrizSup.append(PPV)

    #False discovery rate (FDR):
    FDR = FP/PP
    supportName.append("FDR")
    matrizSup.append(FDR)

    #False omission rate (FOR):
    FOR = FN/PN
    supportName.append("FOR")
    matrizSup.append(FOR)

    #Negative predictive value (NPV):
    NPV = TN/PN
    supportName.append("NPV")
    matrizSup.append(NPV)

    #True positive rate (TPR), recall, sensitivity (SEN):
    TPR = TP/AP
    supportName.append("TPR")
    matrizSup.append(TPR)

    #False positive rate (FPR):
    FPR = FP/(TN + FP)
    supportName.append("FPR")
    matrizSup.append(FPR)

    #Prevalence threshold (PT):
    PT = ((TPR*FPR)**(1/2) - FPR)/(TPR - FPR)
    supportName.append("PT")
    matrizSup.append(PT)

    #False negative rate (FNR):
    FNR = FN/AP
    supportName.append("FNR")
    matrizSup.append(FNR)

    #True negative rate (TNR):
    TNR = TN/AN
    supportName.append("TNR")
    matrizSup.append(TNR)

    #Positive likelihood ratio (LRp):
    LRp = TPR/FPR
    supportName.append("LRp")
    matrizSup.append(LRp)

    #Negative likelihood ratio (LRn):
    LRn = FNR/TNR
    supportName.append("LRn")
    matrizSup.append(LRn)

    #Markedness (MK):
    MK = PPV/NPV
    supportName.append("MK")
    matrizSup.append(MK)

    #Diagnostic odds ratio (DOR):
    DOR = LRp/LRn
    supportName.append("DOR")
    matrizSup.append(DOR)

    #Balanced accuracy (BA):
    BA = TPR/TNR
    supportName.append("BA")
    matrizSup.append(BA)

    #F1 score:
    F1 = (2*TP)/(2*TP+FP+FN)
    supportName.append("F1 score")
    matrizSup.append(F1)

    #Fowlkes–Mallows index (FM):
    FM = (PPV*TPR)**(1/2)
    supportName.append("FM")
    matrizSup.append(FM)

    #Matthews correlation coefficient (MCC):
    MCC = (TPR*TNR*PPV*NPV)**(1/2) - (FNR*FPR*FOR*FDR)**(1/2)
    supportName.append("MCC")
    matrizSup.append(MCC)

    #Threat score (TS), critical success index (CSI):
    TS = TP/(TP+FN+FP)
    supportName.append("TS")
    matrizSup.append(TS)

    return np.array(matrizSup), supportName

def areaTrap(x1, x2, y1, y2):
    base = abs(x1 - x2)
    h = (y1+y2)/2
    return base*h

def ROC(Y_test, p, label):
    fpr = []
    tpr = []
    FP = 0
    TP = 0
    A = 0
    FP_prev = 0
    TP_prev = 0
    f_prev = -1
    L = Y_test
    f = p
    N = 0
    P = 0
    for ex in L:
        if ex == 0:
            N += 1
        else:
            P += 1
    f, L = organizador(f, L)
    L = L[::-1]
    f = f[::-1]
    for j in range(len(L)):
        if f[j] != f_prev:
            fpr.append(FP/N)
            tpr.append(TP/P)
            A += areaTrap(FP, FP_prev, TP, TP_prev)
            f_prev = f[j]
            FP_prev = FP
            TP_prev = TP
        if L[j] == 1:
            TP += 1
        else:
            FP += 1
    fpr.append(FP/N)
    tpr.append(TP/P)
    A += areaTrap(N, FP_prev, P, TP_prev)
    A = A/(P*N)

    fig = plt.figure()
    plt.plot([0, 1], [0, 1], linestyle='dashed', color="k")
    plt.plot(fpr, tpr, label=f'ROC Curve (area = {A})', color="tab:orange")
    plt.legend()
    plt.grid()
    plt.ylabel("True Positeve Rate")
    plt.xlabel("False Positive Rate")
    plt.title(f'Receiver Operating Characteristic for Class {label}')

    return fig

# Importação do dataset

iris = datasets.load_iris()

# Nesse banco de dados, os principais valores que iremos utilizar serão as entradas (iris.data) e as saídas (iris.target)

X = iris.data

Y = iris.target

# Obtendo as classes existentes para classificação:

classes = list(set(Y))

# Modelos one x rest:

oneXrest = []

for c in classes:
    oneXrest.append(np.array(list(map(lambda x: 1 if x == c else 0 , Y))))

# Modelo Multiclasse:

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

pred, p = knn(X_train, Y_train, X_test)
    
print(p)
print(f'Teste: {Y_test}\nPred:  {np.array(pred)}')

# Matriz Confusão:

matrizConf = matrizConfusao(pred, Y_test)
dfMatrizConf = pd.DataFrame(matrizConf, columns=classes)
dfMatrizConf.index = classes

print(dfMatrizConf)

# Matriz Suporte:

matrizSup, supportNames = matrizSuporte(matrizConf)
dfMatrizSup = pd.DataFrame(matrizSup.T, columns=supportNames)
dfMatrizSup.index = classes

print(dfMatrizSup)

# Modelo One X Rest:
fig = []

for i in range(len(oneXrest)):
    print(f'---------------------------------------------------\nClasse de referência: {classes[i]}')

    y = oneXrest[i]
    oxrclasses = list(set(y))

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

    pred, p = knn(X_train, Y_train, X_test)
    
    p = np.array(p)
    print(p)
    print(f'Teste: {Y_test}\nPred:  {np.array(pred)}')

    # Matriz Confusão:

    matrizConf = matrizConfusao(pred, Y_test)
    dfMatrizConf = pd.DataFrame(matrizConf, columns=oxrclasses)
    dfMatrizConf.index = oxrclasses

    print(dfMatrizConf)

    # Matriz Suporte:

    matrizSup, supportNames = matrizSuporte(matrizConf)
    dfMatrizSup = pd.DataFrame(matrizSup.T, columns=supportNames)
    dfMatrizSup.index = oxrclasses

    print(dfMatrizSup)

    # Curva ROC:

    fig.append(ROC(Y_test, p, str(i)))

plt.show()