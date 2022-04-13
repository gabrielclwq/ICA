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
from sklearn.preprocessing import label_binarize
from itertools import cycle

"""svm.smo.py: Support Vector Machine implementation with smo"""

__author__ = "Gabriel Costa Leite"
__email__ = "gabriel.wq@alu.ufc.br"
__maintainer__ = "Gabriel Costa Leite"
__status__ = "Production"

class KNNAnalytics():

    def __init__(self, X, Y, kn=3, r=2, test_size=0.3):
        self.X = X
        self.Y = Y
        self.testsize = test_size
        self.classes = list(set(self.Y))
        self.kn = kn
        self.r = r
        self.test_size = test_size
        self.supportName = []
        self.matrizSup = []
        self.matrizConf = self.matrizConfusao()

        self.oneXrest = []

        for c in self.classes:
            self.oneXrest.append(list(map(lambda x: 1 if x == c else 0 , self.Y)))

        self.pred, self.p = self.knn()

    def modelo_multiclass(self):
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, self.testsize)

    def modelo_oneXrest(self):


    def dMinkowski(self, p, q, r):
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

    def knn(self, X_train, Y_train, X_test):
        pred = []
        p = []
        for i in self.X_test:
            distances = []
            index = []
            for j in range(len(self.X_train)):
                distances.append(self.dMinkowski(i, self.X_train[j], self.r))
                index.append(j)
            distances, index = self.organizador(distances, index)
            distances = np.array(distances[0:self.kn])
            index = np.array(index[0:self.kn])
            y_train = np.array(self.Y_train[index])
            m = ((distances*y_train).sum())/(distances.sum())
            p.append(m)
            pred.append(round(m))
        return pred, p

    def matrizConfusao(self, Y_test, pred):
        #Obtendo a matriz de confusão:
        y_set = self.classes
        matrizConf = np.zeros((len(y_set), len(y_set)))
        for i in range(len(y_set)): #i -> real
            for j in range(len(y_set)): #j -> previsao
                nReal = 0 #total
                nPrev = 0 #acertos
                for k in range(len(Y_test)):
                    if self.Y_test[k] == y_set[i]:
                        nReal += 1
                        if pred[k] == y_set[j]:
                            nPrev += 1
                matrizConf[i,j] = nPrev
        return matrizConf

    def matrizSuporte(self):
        supportName = []
        matrizSup = []

        #Positive Prediction:
        PP = []
        for i in range(len(self.matrizConf[0])):
            PP.append(self.matrizConf[:,i].sum())
        supportName.append("PP")
        PP = np.array(PP)
        matrizSup.append(PP)

        #Negative Prediction:
        PN = []
        for i in range(len(self.matrizConf[0])):
            PN.append(self.matrizConf.sum() - self.matrizConf[:,i].sum())
        supportName.append("PN")
        PN = np.array(PN)
        matrizSup.append(PN)

        #Actual Positive:
        AP = []
        for i in range(len(self.matrizConf[0])):
            AP.append(self.matrizConf[i].sum())
        supportName.append("AP")
        AP = np.array(AP)
        matrizSup.append(AP)

        #Actual Negative:
        AN = []
        for i in range(len(self.matrizConf[0])):
            AN.append(self.matrizConf.sum() - self.matrizConf[i].sum())
        supportName.append("AN")
        AN = np.array(AN)
        matrizSup.append(AN)

        #True positive:
        TP = []
        for i in range(len(self.matrizConf[0])):
            TP.append(self.matrizConf[i,i])
        supportName.append("TP")
        TP = np.array(TP)
        matrizSup.append(TP)

        #False positive:
        FP = []
        for i in range(len(self.matrizConf[0])):
            FP.append(self.matrizConf[:,i].sum() - self.matrizConf[i,i])
        supportName.append("FP")
        FP = np.array(FP)
        matrizSup.append(FP)

        #False positive:
        FN = []
        for i in range(len(self.matrizConf[0])):
            FN.append(self.matrizConf[i].sum() - self.matrizConf[i,i])
        supportName.append("FN")
        FN = np.array(FN)
        matrizSup.append(FN)

        #True negative:
        TN = []
        for i in range(len(self.matrizConf[0])):
            TN.append(self.matrizConf.sum() - self.matrizConf[i].sum() - self.matrizConf[:,i].sum() + self.matrizConf[i,i])
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

        return np.array(self.matrizSup), supportName

    def 