from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score, f1_score
import numpy as np

def scores(model, dataset, threshold=None):
    Xc = dataset.X_nt[dataset.test_idx]
    Xe = dataset.Xeq_nt[dataset.test_idx]
    Y = dataset.Y[dataset.test_idx]
    if threshold is None:
        # threshold = sum(model.expectation(dataset.X_nt[dataset.train_idx], dataset.Xeq_nt[dataset.train_idx], dataset.Y[dataset.train_idx]))/len(dataset.train_idx)
        threshold = np.mean(dataset.Y[dataset.train_idx])
    perfs = {}
    perfs['ROC AUC'] = roc_auc_score(Y, model.predict_proba(Xc, Xe))
    perfs['PR AUC'] = average_precision_score(Y, model.predict_proba(Xc, Xe))
    perfs['Recall'] = recall_score(Y, model.predict(Xc, Xe, threshold))
    perfs['Precision'] = precision_score(Y, model.predict(Xc, Xe, threshold))
    perfs['F1'] = f1_score(Y, model.predict(Xc, Xe, threshold))
    return perfs

def cscores(model, dataset, threshold=None):
    Xc = dataset.X_nt[dataset.test_idx]
    try:
        Z = dataset.data.Z.loc[dataset.test_idx]
    except:
        Z = dataset.Z[dataset.test_idx]
    if threshold is None:
        if dataset.simple:
            threshold = sum(model.expectation(dataset.X_nt[dataset.train_idx], dataset.F[dataset.train_idx][:,np.newaxis], dataset.Y[dataset.train_idx]))/len(dataset.train_idx)
        else:
            threshold = sum(model.expectation(dataset.X_nt[dataset.train_idx], dataset.Xeq_nt[dataset.train_idx], dataset.Y[dataset.train_idx]))/len(dataset.train_idx)
        # print(threshold)
    perfs = {}
    perfs['ROC AUC'] = roc_auc_score(Z, model.predict_cproba(Xc))
    perfs['PR AUC'] = average_precision_score(Z, model.predict_cproba(Xc))
    perfs['Recall'] = recall_score(Z, model.predict_c(Xc, threshold))
    perfs['Precision'] = precision_score(Z, model.predict_c(Xc, threshold))
    perfs['F1'] = f1_score(Z, model.predict_c(Xc, threshold))
    return perfs

def base_scores(model, dataset, threshold=None):
    Xe = dataset.Xeq_nt[dataset.test_idx]
    Y = dataset.Y[dataset.test_idx]
    if threshold is None:
        threshold = np.mean(dataset.Y[dataset.train_idx])
    perfs = {}
    perfs['ROC AUC'] = roc_auc_score(Y, model.predict_cont(Xe))
    perfs['PR AUC'] = average_precision_score(Y, model.predict_cont(Xe))
    perfs['Recall'] = recall_score(Y, model.predict_cont(Xe)>=threshold)
    perfs['Precision'] = precision_score(Y, model.predict_cont(Xe)>=threshold)
    perfs['F1'] = f1_score(Y, model.predict_cont(Xe)>=threshold)
    return perfs

def base_cscores(model, dataset, threshold=None):
    Xe = dataset.X_nt[dataset.test_idx,:]
    Z = dataset.data.Z.loc[dataset.test_idx]
    if threshold is None:
        threshold = np.mean(model.predict_cont(dataset.X_nt[dataset.train_idx]))
        # print(threshold)
    perfs = {}
    perfs['ROC AUC'] = roc_auc_score(Z, model.predict_cont(Xe))
    perfs['PR AUC'] = average_precision_score(Z, model.predict_cont(Xe))
    perfs['Recall'] = recall_score(Z, model.predict_cont(Xe)>=threshold)
    perfs['Precision'] = precision_score(Z, model.predict_cont(Xe)>=threshold)
    perfs['F1'] = f1_score(Z, model.predict_cont(Xe)>=threshold)
    return perfs

def DV_scores(dataset, threshold=1.):
    preds_cont = dataset.predict_DV_crack()
    preds = (preds_cont >= threshold).astype(int)
    Y = dataset.Y[dataset.test_idx]
    perfs = {}
    perfs['ROC AUC'] = roc_auc_score(Y, preds_cont)
    perfs['PR AUC'] = average_precision_score(Y, preds_cont)
    perfs['Recall'] = recall_score(Y, preds)
    perfs['Precision'] = precision_score(Y, preds)
    perfs['F1'] = f1_score(Y, preds)
    return perfs   

def DV_cscores(dataset, threshold=1.):
    preds_cont = dataset.predict_DV_dim()
    preds = (preds_cont >= threshold).astype(int)
    Z = dataset.data.Z.iloc[dataset.test_idx]
    perfs = {}
    perfs['ROC AUC'] = roc_auc_score(Z, preds_cont)
    perfs['PR AUC'] = average_precision_score(Z, preds_cont)
    perfs['Recall'] = recall_score(Z, preds)
    perfs['Precision'] = precision_score(Z, preds)
    perfs['F1'] = f1_score(Z, preds)
    return perfs    