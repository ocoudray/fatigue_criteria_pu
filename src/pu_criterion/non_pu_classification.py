from sklearn.metrics import roc_auc_score, average_precision_score, hinge_loss
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from sklearn.inspection import permutation_importance

class Classifier:
    def __init__(self, model):
        self.model = model
    
    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)
    
    def set_param(self, name, value):
        self.model.__setattr__(name, value)
    
    def predict_cont(self, X_test):
        return self.model.predict_proba(X_test)[:,1]
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def score(self, X_test, Y_test, scoring='roc'):
        if scoring=='roc':
            return roc_auc_score(Y_test, self.predict_cont(X_test))
        elif scoring=='ap':
            return average_precision_score(Y_test, self.predict_cont(X_test))
        elif scoring=='hinge':
            return hinge_loss(Y_test, self.predict_cont(X_test))
        else:
            print('Unknown scoring function')

class LogRegression(Classifier):
    def __init__(self, penalty='none', class_weight='balanced'):
        self.hyper_params = 10**np.linspace(-4,0,20)
        return super().__init__(LogisticRegression(penalty=penalty, class_weight=class_weight))
    def set_hyper_param(self, val):
        self.set_param('C', val)

class LinearSVM(Classifier):
    def __init__(self, class_weight='balanced', C=1.):
        self.hyper_params = 10**np.linspace(-6,0,20)
        return super().__init__(LinearSVC(class_weight=class_weight, C=C, loss='hinge'))
    def predict_cont(self, X_test):
        return self.model.decision_function(X_test)
    def set_hyper_param(self, val):
        self.set_param('C', val)

class GaussianKernelSVM(Classifier):
    def __init__(self, class_weight='balanced', C=1.):
        self.hyper_params = 10**np.linspace(-4,0,20)
        return super().__init__(SVC(class_weight=class_weight, C=C))
    def fit(self, X_train, Y_train):
        # self.scaler = StandardScaler()
        # self.scaler.fit(X_train)
        # return super().fit(self.scaler.transform(X_train), Y_train)
        return super().fit(X_train, Y_train)
    def predict_cont(self, X_test):
        # return self.model.decision_function(self.scaler.transform(X_test))
        return self.model.decision_function(X_test)
    def set_hyper_param(self, val):
        self.set_param('gamma', val)

class RandomForest(Classifier):
    def __init__(self, penalty='l1', class_weight='balanced', depth=5):
        self.hyper_params = np.arange(1, 20, 1)
        return super().__init__(RandomForestClassifier(max_depth=depth, class_weight=class_weight))
        # return super().__init__(RandomForestClassifier(min_samples_leaf=10, max_depth=5, class_weight=class_weight))
    def set_hyper_param(self, val):
        self.set_param('max_depth', val)
        # self.set_param('min_samples_leaf', val)
    # def predict_cont(self, X_test):
    #     return np.mean(np.array([estimator.predict(X_test) for estimator in self.model.estimators_]), axis=0)
    def feature_importance(self, X, Y):
        # importances = self.model.feature_importances_
        # std = np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)
        result = permutation_importance(self.model, X, Y, n_repeats=5, random_state=42, n_jobs=2, scoring=['roc_auc', 'average_precision'])
        return result['roc_auc'].importances.T, result['average_precision'].importances.T

class LDA(Classifier):
    def __init__(self):
        self.hyper_params = [1]
        return super().__init__(LinearDiscriminantAnalysis())
    def set_hyper_param(self, val):
        pass