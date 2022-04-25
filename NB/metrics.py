import numpy as np

def matrizConfusao(Y_test, pred):
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

def accuracy(Y_test, pred):
    total = 0
    acerto = 0
    for i in range(len(Y_test)):
        if pred[i] == Y_test[i]:
            acerto += 1
        total += 1
    accuracy = acerto/total
    return accuracy

def MSE(Y_test, pred):
    e = 0
    for i in range(len(Y_test)):
        e += (pred[i] - Y_test[i])**2
    e = e/len(Y_test)
    return e

def ROC(Y_test, scores):

    def areaTrap(x1, x2, y1, y2):
        base = abs(x1 - x2)
        h = (y1+y2)/2
        return base*h

    def organizador(my_list1, my_list2):
        for i in range(len(my_list1)):
            for j in range(i + 1, len(my_list1)):
                if my_list1[i] > my_list1[j]:
                    my_list1[i], my_list1[j] = my_list1[j], my_list1[i]
                    my_list2[i], my_list2[j] = my_list2[j], my_list2[i]
        return(my_list1, my_list2)

    fpr = []
    tpr = []
    FP = 0
    TP = 0
    A = 0
    FP_prev = 0
    TP_prev = 0
    f_prev = -1
    L = Y_test
    f = scores
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

    return fpr, tpr, A