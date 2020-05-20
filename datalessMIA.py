import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


#evaluation will be elsewhere, because it is shared across every attacker
class DatalessMIA:

    def __init__(self, statisticalOperator, targetModel,  dataGenerator, top=90):
        self.statisticalOperator = statisticalOperator
        self.targetModel = targetModel
        self.dataGenerator = dataGenerator
        self.top = top
        self.threshold = None

#determines the threshold
    def fit(self, X):
        posterior = self.targetModel.predict(self.dataGenerator(X))
        stats = self.statisticalOperator(posterior)
        #TODO unsure about numpys percentile function
        self.threshold = np.percentile(stats, self.top)

    def predict_proba(self, X):
        return self.statisticalOperator(self.targetModel.predict(X))

    def predict(self, X):
        return self.predict_proba(X) >= self.threshold

    def fit_predict_proba(self, X):
        self.fit(X)
        return self.predict_proba(X)


