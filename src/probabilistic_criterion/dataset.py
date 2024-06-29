import numpy as np
import pandas as pd

def singularity(tau0):
    if tau0 == 93.:
        return 'W'
    elif tau0 == 104.55 or tau0 == 127.5 or tau0 == 96.9 or tau0 == 99.45 or tau0 == 105.4 or tau0 == 141.1:
        return 'E'
    else:
        return 'P'

class FDB:
    def __init__(self, path, cfeatures, efeatures, feature_norm='Tau0_1', ctransform = lambda x : x, etransform = lambda x : x):
        self.ctransform = ctransform
        self.etransform = etransform
        features = list(set(cfeatures).union(efeatures))
        data = pd.read_csv(path, index_col=0)
        data = data[(data.Epr==False)&(data.fn>0)&(data['Tau0_0_mean']<=150)]
        data['label'] = data.Model + '__' + data.Zone
        data['Z'] = data.Zone.apply(lambda val:val[0]=='C').astype('int')
        data['intercept'] = 1.
        data = data[(data.Z==1)|(data.z==0)]
        data = data.loc[data['Tau0_0_mean'].dropna().index]
        data.reset_index(inplace=True)
        data = data.replace(np.nan, 0.)
        data['Zone_type'] = data.Tau0_0_mean.apply(singularity)
        self.col_index_fn = np.where([((len(f.split('_'))>=2) and (f.split('_')[1]=='a')) or (f=='tau') or (f=='p') or (f=='tau_n') or (f=='p_n')  for f in efeatures])[0]
        col_index_norm = np.where([((len(f.split('_'))>=2) and (f.split('_')[-1]=='n'))  for f in features])[0]
        if feature_norm is None:
            feature_norm = 'intercept'
        for i in col_index_norm:
            f = features[i]
            data[f] = data[f[:-2]]/data[feature_norm]
        for f in features:
            if (len(f.split('_'))>=2) and (f.split('_')[1]=='a'):
                data[f] = np.abs(data[f])
        self.data = data
        self.Y = data.y.values
        self.X = data[cfeatures].values
        self.F = data.fm.values
        self.Xeq = data[efeatures].values
        self.simple = False

    def __len__(self):
        return len(self.Y)
    
    def split(self, seed=0, factor=None, cv=None, model=None, upsample=1):
        if model is None:
            labeled = np.unique(self.data[self.data.Z==1].label)
            unlabeled = np.unique(self.data[self.data.Z==0].label)
            ratio = np.mean(self.data.Z)
            np.random.seed(seed)
            np.random.shuffle(labeled)
            np.random.shuffle(unlabeled)
            n1, n0 = len(labeled), len(unlabeled)
            if factor is None:
                n_u = n0
            else:
                n_u = int(n1*factor)
            if cv is None:
                train_lab_idx, test_lab_idx, val_lab_idx = list(np.arange(0,n1//2)), list(np.arange(n1//2, n1)), []
                train_unl_idx, test_unl_idx, val_unl_idx = list(np.arange(0,n_u//2)), list(np.arange(n0//2, n0)), []
            else:
                n_splits, it = cv
                train_lab_idx, test_lab_idx, val_lab_idx = list(np.arange(0,it*n1//(2*n_splits))) + list(np.arange((it+1)*n1//(2*n_splits), n1//2)), list(np.arange(n1//2, n1)), list(np.arange(it*n1//(2*n_splits), (it+1)*n1//(2*n_splits)))
                train_unl_idx, test_unl_idx, val_unl_idx = list(np.arange(0,it*n_u//(2*n_splits))) + list(np.arange((it+1)*n_u//(2*n_splits), n_u//2)), list(np.arange(n_u//2, n_u)), list(np.arange(it*n_u//(2*n_splits), (it+1)*n_u//(2*n_splits)))
            self.train_idx = list(self.data[self.data.label.isin(labeled[train_lab_idx])].index)
            self.train_idx = list(np.repeat(self.train_idx, int(upsample)))
            self.test_idx = list(self.data[self.data.label.isin(labeled[test_lab_idx])].index)
            self.val_idx = list(self.data[self.data.label.isin(labeled[val_lab_idx])].index)
            self.train_idx += list(self.data[(self.data.label.isin(unlabeled[train_unl_idx]))&(~self.data.label.isin(labeled))].index)
            self.test_idx += list(self.data[(self.data.label.isin(unlabeled[test_unl_idx]))&(~self.data.label.isin(labeled))].index)
            self.val_idx += list(self.data[(self.data.label.isin(unlabeled[val_unl_idx]))&(~self.data.label.isin(labeled))].index)
        else:
            self.train_idx = list(self.data[self.data.Model != model].index)
            self.test_idx = list(self.data[self.data.Model == model].index)


    def prepare(self, random_state=0, factor=None, cv=None, model=None, upsample=1):
        self.split(random_state, factor, cv=cv, model=model, upsample=upsample)
        cmean, cstd = self.X[self.train_idx].mean(axis=0), self.X[self.train_idx].std(axis=0)
        self.X_n = (self.X-cmean) / cstd
        emean, estd = self.Xeq[self.train_idx].mean(axis=0), self.Xeq[self.train_idx].std(axis=0)
        self.Xeq_n = self.Xeq / estd
        for j in self.col_index_fn:
            self.Xeq_n[:,j] = self.Xeq_n[:,j]*self.F
            self.Xeq[:,j] = self.Xeq[:,j]*self.F
        self.X_nt = self.ctransform(self.X_n)
        self.Xeq_nt = self.etransform(self.Xeq_n)
    
    def get_CD_norm(self, X):
        estd = self.Xeq[self.train_idx].std(axis=0)
        return 0.35*estd[0]*X[:,0] + estd[1]*X[:,1]

        
    def get_train(self):
        if self.train_idx is not None:
            return self.__getitem__(self.train_idx)
        
    def get_test(self):
        if self.test_idx is not None:
            return self.__getitem__(self.test_idx)

    def get_val(self):
        if self.val_idx is not None:
            return self.__getitem__(self.val_idx)

    def __getitem__(self, idx):
        x = self.X_nt[idx]
        xeq = self.Xeq_nt[idx]
        label = self.Y[idx]
        z = self.data.Z.loc[idx]
        return x, xeq, z, label
    
    def predict_DV_dim(self, testing=True):
        DV_pred = (0.35 * self.data.p + self.data.tau)/self.data['Tau0_0_mean']
        if testing:
            return DV_pred.iloc[self.test_idx].values
        else:
            return DV_pred.values

    def predict_DV_crack(self, testing=True):
        DV_pred = self.data.fn*(0.35 * self.data.p + self.data.tau)/self.data['Tau0_0_mean']
        if testing:
            return DV_pred.iloc[self.test_idx].values
        else:
            return DV_pred.values