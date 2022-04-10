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

"""svm.smo.py: Support Vector Machine implementation with smo"""

__author__ = "Gabriel Costa Leite"
__email__ = "gabriel.wq@alu.ufc.br"
__maintainer__ = "Gabriel Costa Leite"
__status__ = "Production"

class SVMAnalytics():

    def __init__(self, X, Y, C=1, gamma=1, test_size=0.3):
        self.X = X
        self.Y = Y
        self.Xnames = []
        self.Ynames = []
        self.supportName = []
        self.C = C
        self.gamma = gamma
        self.test_size = test_size
        self.clf = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.pred = None
        self.matrizConf = None
        self.matrizSup = []

        self.preprocessing()

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=self.test_size)
        self.clf = SVC(C=self.C, gamma=self.gamma)
        self.clf.fit(self.X_train, self.Y_train)
        self.pred = self.clf.predict(self.X_test)

        self.matrizConfusao()
        self.matrizSuporte()

    def preprocessing(self):
        #Transformando string em int:
        X_set = []
        for i in self.X.columns:
            x_set = list(set(self.X[i]))
            a = 1
            for j in x_set:
                self.X[i] = list(map(lambda x: a if x == j else x , self.X[i]))
                a += 1
            X_set.append(x_set)

        Y_set = []
        y_set = list(set(self.Y))
        a = 1
        for j in y_set:
            self.Y = list(map(lambda x: a if x == j else x , self.Y))
            a += 1
        Y_set.append(y_set)

        """ #Normalizando X:
        for i in self.X.columns:
            self.X[i] = self.X[i]/max(self.X[i])
 """
        self.Ynames = Y_set[0]
        self.Xnames = X_set

    def matrizConfusao(self):
        #Obtendo a matriz de confusão:
        y_set = list(set(self.Y_test))
        self.matrizConf = np.zeros((len(y_set), len(y_set)))
        for i in range(len(y_set)): #i -> real
            for j in range(len(y_set)): #j -> previsao
                nReal = 0 #total
                nPrev = 0 #acertos
                for k in range(len(self.Y_test)):
                    if self.Y_test[k] == y_set[i]:
                        nReal += 1
                        if self.pred[k] == y_set[j]:
                            nPrev += 1
                self.matrizConf[i,j] = nPrev

    def matrizSuporte(self):
        #Positive Prediction:
        PP = []
        for i in range(len(self.matrizConf[0])):
            PP.append(self.matrizConf[:,i].sum())
        self.supportName.append("PP")
        PP = np.array(PP)
        self.matrizSup.append(PP)

        #Negative Prediction:
        PN = []
        for i in range(len(self.matrizConf[0])):
            PN.append(self.matrizConf.sum() - self.matrizConf[:,i].sum())
        self.supportName.append("PN")
        PN = np.array(PN)
        self.matrizSup.append(PN)


        #Actual Positive:
        AP = []
        for i in range(len(self.matrizConf[0])):
            AP.append(self.matrizConf[i].sum())
        self.supportName.append("AP")
        AP = np.array(AP)
        self.matrizSup.append(AP)

        #Actual Negative:
        AN = []
        for i in range(len(self.matrizConf[0])):
            AN.append(self.matrizConf.sum() - self.matrizConf[i].sum())
        self.supportName.append("AN")
        AN = np.array(AN)
        self.matrizSup.append(AN)

        #True positive:
        TP = []
        for i in range(len(self.matrizConf[0])):
            TP.append(self.matrizConf[i,i])
        self.supportName.append("TP")
        TP = np.array(TP)
        self.matrizSup.append(TP)

        #False positive:
        FP = []
        for i in range(len(self.matrizConf[0])):
            FP.append(self.matrizConf[:,i].sum() - self.matrizConf[i,i])
        self.supportName.append("FP")
        FP = np.array(FP)
        self.matrizSup.append(FP)

        #False positive:
        FN = []
        for i in range(len(self.matrizConf[0])):
            FN.append(self.matrizConf[i].sum() - self.matrizConf[i,i])
        self.supportName.append("FN")
        FN = np.array(FN)
        self.matrizSup.append(FN)

        #True negative:
        TN = []
        for i in range(len(self.matrizConf[0])):
            TN.append(self.matrizConf.sum() - self.matrizConf[i].sum() - self.matrizConf[:,i].sum() + self.matrizConf[i,i])
        self.supportName.append("TN")
        TN = np.array(TN)
        self.matrizSup.append(TN)

        #Prevalence:
        prevalence = AP/(AP+AN)
        self.supportName.append("Prevalence")
        self.matrizSup.append(prevalence)

        #Accuracy:
        accuracy = (TP+TN)/(AP+AN)
        self.supportName.append("Accuracy")
        self.matrizSup.append(accuracy)

        #Positive predictive value (PPV):
        PPV = TP/PP
        self.supportName.append("PPV")
        self.matrizSup.append(PPV)

        #False discovery rate (FDR):
        FDR = FP/PP
        self.supportName.append("FDR")
        self.matrizSup.append(FDR)

        #False omission rate (FOR):
        FOR = FN/PN
        self.supportName.append("FOR")
        self.matrizSup.append(FOR)

        #Negative predictive value (NPV):
        NPV = TN/PN
        self.supportName.append("NPV")
        self.matrizSup.append(NPV)

        #True positive rate (TPR), recall, sensitivity (SEN):
        TPR = TP/AP
        self.supportName.append("TPR")
        self.matrizSup.append(TPR)

        #False positive rate (FPR):
        FPR = FP/AN
        self.supportName.append("FPR")
        self.matrizSup.append(FPR)

        #Prevalence threshold (PT):
        PT = ((TPR*FPR)**(1/2) - FPR)/(TPR - FPR)
        self.supportName.append("PT")
        self.matrizSup.append(PT)

        #False negative rate (FNR):
        FNR = FN/AP
        self.supportName.append("FNR")
        self.matrizSup.append(FNR)

        #True negative rate (TNR):
        TNR = TN/AN
        self.supportName.append("TNR")
        self.matrizSup.append(TNR)

        #Positive likelihood ratio (LRp):
        LRp = TPR/FPR
        self.supportName.append("LRp")
        self.matrizSup.append(LRp)

        #Negative likelihood ratio (LRn):
        LRn = FNR/TNR
        self.supportName.append("LRn")
        self.matrizSup.append(LRn)

        #Markedness (MK):
        MK = PPV/NPV
        self.supportName.append("MK")
        self.matrizSup.append(MK)

        #Diagnostic odds ratio (DOR):
        DOR = LRp/LRn
        self.supportName.append("DOR")
        self.matrizSup.append(DOR)

        #Balanced accuracy (BA):
        BA = TPR/TNR
        self.supportName.append("BA")
        self.matrizSup.append(BA)

        #F1 score:
        F1 = (2*TP)/(2*TP+FP+FN)
        self.supportName.append("F1 score")
        self.matrizSup.append(F1)

        #Fowlkes–Mallows index (FM):
        FM = (PPV*TPR)**(1/2)
        self.supportName.append("FM")
        self.matrizSup.append(FM)

        #Matthews correlation coefficient (MCC):
        MCC = (TPR*TNR*PPV*NPV)**(1/2) - (FNR*FPR*FOR*FDR)**(1/2)
        self.supportName.append("MCC")
        self.matrizSup.append(MCC)

        #Threat score (TS), critical success index (CSI):
        TS = TP/(TP+FN+FP)
        self.supportName.append("TS")
        self.matrizSup.append(TS)

        self.matrizSup = np.array(self.matrizSup)

    def getModelo(self):
        return self.clf
    
    def getRes(self):
        return self.pred

    def getX_test(self):
        return self.X_test

    def getX_train(self):
        return self.X_train

    def getY_train(self):
        return self.Y_train

    def getY_test(self):
        return self.Y_test

    def getNames(self):
        return self.Xnames, self.Ynames

    def getConstants(self):
        return self.C, self.gamma

    def getMatrizConfusao(self):
        dfMatrizConf = pd.DataFrame(self.matrizConf, columns=self.Ynames)
        dfMatrizConf.index = self.Ynames
        return dfMatrizConf
        
    def getMatrizSuporte(self):
        dfMatrizSup = pd.DataFrame(self.matrizSup.T, columns=self.supportName)
        dfMatrizSup.index = self.Ynames
        return dfMatrizSup

    def getAccuracy(self):
        df = self.getMatrizSuporte()
        return df['Accuracy'].mean()

    def getAllDataClassifiersGraph(self):
        fig, axs = plt.subplots(2)
        fig.suptitle('Data Classifiers')

        def organizador(my_list1, my_list2):
            for i in range(len(my_list1)):
                for j in range(i + 1, len(my_list1)):
                    if my_list1[i] > my_list1[j]:
                        my_list1[i], my_list1[j] = my_list1[j], my_list1[i]
                        my_list2[i], my_list2[j] = my_list2[j], my_list2[i]
            return(my_list1, my_list2)

        Y_, X_ = organizador(self.Y, self.X.values)
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
            axs[1].scatter(X_[lens[i]:lens[i+1]-1, 0], X_.sum(axis=1)[lens[i]:lens[i+1]-1]+i*100)
        axs[0].legend(self.Ynames)
        axs[0].set_title("Real")
        axs[1].set_title("Distribuido")
        for ax in axs.flat:
            ax.set(xlabel='soma de atributos', ylabel='primeiro atributo')
        for ax in axs.flat:
            ax.label_outer()

        return fig

    def getTrainDataClassifiersGraph(self):
        fig, axs = plt.subplots(2)
        fig.suptitle('Train Data Classifiers')

        def organizador(my_list1, my_list2):
            for i in range(len(my_list1)):
                for j in range(i + 1, len(my_list1)):
                    if my_list1[i] > my_list1[j]:
                        my_list1[i], my_list1[j] = my_list1[j], my_list1[i]
                        my_list2[i], my_list2[j] = my_list2[j], my_list2[i]
            return(my_list1, my_list2)

        Y_, X_ = organizador(self.Y_train, self.X_train.values)
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
            axs[1].scatter(X_[lens[i]:lens[i+1]-1, 0], X_.sum(axis=1)[lens[i]:lens[i+1]-1]+i*100)
        axs[0].legend(self.Ynames)
        axs[0].set_title("Real")
        axs[1].set_title("Distribuido")
        for ax in axs.flat:
            ax.set(xlabel='soma de atributos', ylabel='primeiro atributo')
        for ax in axs.flat:
            ax.label_outer()

        return fig

    def getTestDataClassifiersGraph(self):
        fig, axs = plt.subplots(2)
        fig.suptitle('Test Data Classifiers')

        def organizador(my_list1, my_list2):
            for i in range(len(my_list1)):
                for j in range(i + 1, len(my_list1)):
                    if my_list1[i] > my_list1[j]:
                        my_list1[i], my_list1[j] = my_list1[j], my_list1[i]
                        my_list2[i], my_list2[j] = my_list2[j], my_list2[i]
            return(my_list1, my_list2)
        Y_, X_ = organizador(self.Y_test, self.X_test.values)
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
            axs[1].scatter(X_[lens[i]:lens[i+1]-1, 0], X_.sum(axis=1)[lens[i]:lens[i+1]-1]+i*100)
        axs[0].legend(self.Ynames)
        axs[0].set_title("Real")
        axs[1].set_title("Distribuido")
        for ax in axs.flat:
            ax.set(xlabel='soma de atributos', ylabel='primeiro atributo')
        for ax in axs.flat:
            ax.label_outer()

        return fig

    def getPredDataClassifiersGraph(self):
        fig, axs = plt.subplots(2)
        fig.suptitle('Predicted Data Classifiers')

        def organizador(my_list1, my_list2):
            for i in range(len(my_list1)):
                for j in range(i + 1, len(my_list1)):
                    if my_list1[i] > my_list1[j]:
                        my_list1[i], my_list1[j] = my_list1[j], my_list1[i]
                        my_list2[i], my_list2[j] = my_list2[j], my_list2[i]
            return(my_list1, my_list2)

        Y_, X_ = organizador(self.pred, self.X_test.values)
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
            axs[1].scatter(X_[lens[i]:lens[i+1]-1, 0], X_.sum(axis=1)[lens[i]:lens[i+1]-1]+i*100)
        axs[0].legend(self.Ynames)
        axs[0].set_title("Real")
        axs[1].set_title("Distribuido")
        for ax in axs.flat:
            ax.set(xlabel='soma de atributos', ylabel='primeiro atributo')
        for ax in axs.flat:
            ax.label_outer()

        return fig