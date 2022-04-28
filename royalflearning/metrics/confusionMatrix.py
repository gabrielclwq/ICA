"""
Computes the confusion matrix and all statistics analyses from it

"""

import numpy as np
import pandas as pd

def confusionMatrix(Y_test, pred):

    """Computes the confusion matrix.

        Args:
            Y_test (np.array): Array of real values of classes.
            pred (np.array): Array of predict values of classes.
        
        Return:
            dfMatrizConf (pd.DataFrame): confusion matrix.

    """

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

    dfMatrizConf = pd.DataFrame(matrizConf, columns=y_set())
    dfMatrizConf.index = y_set()

    return dfMatrizConf

def matrizSuporte(matrizConf):

    """Computes the support matrix with all statistics analyses obtained from a confusion matrix.

        Args:
            matrizConf (pd.DataFrame): confusion matrix.
        Return:
            matrizSup (np.array): support matrix.
            supportName (list): list of statistics analyses names computed.

    """
    getTargetLabel = matrizConf.columns.values

    matrizConf = matrizConf.to_numpy()

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

    dfMatrizSup = pd.DataFrame(matrizSup.T, columns=supportName)
    dfMatrizSup.index = getTargetLabel()

    return dfMatrizSup
