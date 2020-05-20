

import numpy as np
import sys
import os
from dataGenerator import dataGeneratorSimple
from datalessMIA import DatalessMIA
from keras.models import load_model
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score




def loadDS(filename):
    with np.load(filename) as data:
        return data['arr_0'], data['arr_1']


def qconc(a,b):
    return np.concatenate((a,b))



statisticalOperator = lambda t : np.max(t, axis=1)
#dut = load_model('kerasCifar10model_500_64')
dut = load_model('kerasCifar10model_500_64')
xin, yin = loadDS('cifarIN.npz')
xout, yout = loadDS('cifarOUT.npz')
atk = DatalessMIA(statisticalOperator, dut, dataGeneratorSimple)
y = qconc(np.ones(30000), np.zeros(30000))
pred = atk.fit_predict_proba((qconc(xin,xout)))
auc = roc_auc_score(y, pred)
ypred = atk.predict((qconc(xin,xout)))
prec = precision_score(y, ypred)
recall = recall_score(y, ypred)
acc = accuracy_score(y, ypred)
print(auc, prec, recall, acc)




