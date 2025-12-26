import numpy as np

class SoftVotingEnsemble:
    def __init__(self, models):
        self.models = models
    
    def fit(self, X, y):
        for name, m in self.models.items():
            m.fit(X, y)
    
    def predict(self, X):
        preds = []
        for name, m in self.models.items():
            preds.append(m.predict(X))

        preds = np.array(preds)

        final = []
        for i in range(preds.shape[1]):
            values, counts = np.unique(preds[:, i], return_counts=True)
            final.append(values[np.argmax(counts)])
        return final
