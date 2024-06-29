from pu_criterion.PU import PUClassifier
from pu_criterion.dataset import FDB
import numpy as np
from pu_criterion.scoring import scores, DV_scores, cscores, DV_cscores, base_scores, base_cscores
from pu_criterion.non_pu_classification import LogRegression
import os
import json

# Adapt the following line with the correct path to csv database file
DATABASE = os.environ["DATABASE_PATH"]

class ExperimentPU:
    def __init__(self, cfeatures, efeatures, random_state, factor, classifier, propensity, balance=False, baseline_model=LogRegression, da=False, model=None, w=1, upsample=1):
        self.cfeatures = cfeatures
        self.efeatures = efeatures
        self.random_state = random_state
        self.factor = factor
        self.model = PUClassifier(classifier(), propensity(), pdf=False, da=da)
        self.model_lr = LogRegression(penalty='none')
        self.dataset = FDB(DATABASE, self.cfeatures, self.efeatures, feature_norm='Tau0_0_mean')
        self.dataset.prepare(random_state=self.random_state, factor=self.factor, model=model, upsample=upsample)
        self.dataset2 = FDB(DATABASE, self.cfeatures, self.cfeatures, feature_norm='Tau0_0_mean')
        self.dataset2.prepare(random_state=self.random_state, factor=self.factor, model=model, upsample=upsample)
        self.X_train, self.Xeq_train, self.Z_train, self.Y_train = self.dataset.get_train()
        self.X_train2, self.Xeq_train2, self.Z_train2, self.Y_train2 = self.dataset2.get_train()
        self.X_test, self.Xeq_test, self.Z_test, self.Y_test = self.dataset.get_test()
        self.X_test2, self.Xeq_test2, self.Z_test2, self.Y_test2 = self.dataset2.get_test()
        self.balance = balance
        self.baseline_model=baseline_model()
        if w=='balanced':
            n0, n1 = np.sum(self.Y_train), len(self.Y_train)-np.sum(self.Y_train)
            w0, w1 = (n0+n1)/(2*n0), (n0+n1)/(2*n1)
            self.w = w0*self.Y_train + w1*(1-self.Y_train)
        else:
            self.w=1
    
    def find_best_init(self, n_inits=20, max_iter=100, verbose=True):
        # Multiple runs to find the best initialization
        self.t1, self.t2 = None, None
        ll = -np.inf
        for k in range(n_inits):
            self.model.initialization(self.X_train, self.Xeq_train, self.Y_train, self.Y_train)
            try:
                self.model.fit(self.X_train, self.Xeq_train, self.Y_train, max_iter=max_iter, warm_start=True, balance=self.balance, w=self.w)
                l = self.model.loglikelihood(self.X_train, self.Xeq_train, self.Y_train, w=self.w)
                if verbose:
                    print('Likelihood at initialization {}: {}'.format(k+1, np.round(l, 3)))
                if l > ll:
                    ll = l
                    self.t1 = self.model.cmodel.params.copy()
                    self.t2 = self.model.emodel.params.copy()
            except:
                print('Exception with initialization no {}'.format(k))

        if verbose:
            print('Best likelihood over {} initializations : {}'.format(n_inits, np.round(ll, 3)))
    
    def fit(self, max_iter=1000, n_inits=20, verbose=True):
        self.find_best_init(n_inits=n_inits, verbose=verbose)
        self.model.cmodel.params = self.t1
        self.model.emodel.params = self.t2
        self.model.fit(self.X_train, self.Xeq_train, self.Y_train, max_iter=100, warm_start=True, balance=self.balance, w=self.w)
    
    def fit_baseline(self):
        self.baseline_model.fit(self.Xeq_train2, self.Y_train2)


    def scores(self):
        self.perfs = scores(self.model, self.dataset)
        self.cscores = cscores(self.model, self.dataset)
        self.base_perfs = base_scores(self.baseline_model, self.dataset2)
        self.base_cscores = base_cscores(self.baseline_model, self.dataset2)
        self.DV_scores = DV_scores(self.dataset)
        self.DV_cscores = DV_cscores(self.dataset)
    
    def save(self, model_path):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self.model.save(os.path.join(model_path, 'model.pkl'))
        np.save(os.path.join(model_path, 'cfeatures.npy'), self.cfeatures)
        np.save(os.path.join(model_path, 'efeatures.npy'), self.efeatures)
        json.dump(self.perfs, open(os.path.join(model_path, 'results.json'), 'w'))
        json.dump(self.cscores, open(os.path.join(model_path, 'cresults.json'), 'w'))
        json.dump(self.base_perfs, open(os.path.join(model_path, 'base_results.json'), 'w'))
        json.dump(self.base_cscores, open(os.path.join(model_path, 'base_cresults.json'), 'w'))
        json.dump(self.DV_scores, open(os.path.join(model_path, 'DVresults.json'), 'w'))
        json.dump(self.DV_cscores, open(os.path.join(model_path, 'DVcresults.json'), 'w'))