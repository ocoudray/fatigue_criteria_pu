import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap

def scatter(dataset, i1, i2, ax, feature_type='dim', idx='train', real_classes=False, s=200):
    if idx == 'train':
        sel_idx = dataset.train_idx
    elif idx == 'test':
        sel_idx = dataset.test_idx
    else:
        sel_idx = np.arange(dataset.X_n.shape[0])
    if feature_type == 'dim':
        X1 = dataset.X_n[sel_idx,i1]
        X2 = dataset.X_n[sel_idx,i2]
    elif feature_type == 'pred':
        X1 = dataset.Xeq_n[sel_idx,i1]
        X2 = dataset.Xeq_n[sel_idx,i2]        
    Y = dataset.Y[sel_idx]
    if real_classes:
        Z = dataset.Z[sel_idx]
        c1, c2 = 'green', 'orange'
    else:
        Z = Y.copy()
        c1, c2 = 'blue', 'red'
    # s = 4*(len(Y)/sum(Y)-1)
    ax.scatter(X1[Z==0], X2[Z==0], color=c1, s=10, label='Y=0', alpha=0.5)
    ax.scatter(X1[Z==1], X2[Z==1], color=c2, s=40, label='Y=1', marker='*')
    ax.legend(fontsize=15)

def criterion_2d(model, dataset, ax, threshold=0.5, real_classes=False):
    hsv_modified = cm.get_cmap('hsv', 256)
    newcmp = ListedColormap(hsv_modified(np.linspace(0.35, 0.1, 256)))
    X = dataset.X_n[dataset.test_idx]
    if real_classes:
        Z = dataset.Z[dataset.test_idx]
    else:
        Z = dataset.Y[dataset.test_idx]
    xmin, xmax = X[:,0].min()-1, X[:,0].max()+1
    ymin, ymax = X[:,1].min()-1, X[:,1].max()+1
    cx = np.linspace(xmin, xmax, 100)
    cy = np.linspace(ymin, ymax, 100)
    CX, CY = np.meshgrid(cx, cy)
    Z = np.zeros(CX.shape)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i,j] = model.eta(dataset.ctransform(np.array([[CX[i,j], CY[i,j]]])))  
    X_train, Xeq_train, Z_train, Y_train = dataset.get_train()
    log_preds = model.eta(X_train)
    ax.contourf(CX, CY, Z, levels=100, label='Fatigue criterion', alpha=0.3, vmin=0, vmax=np.max(log_preds), cmap=newcmp)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r'$P_{c}$', fontsize=25)
    ax.set_ylabel(r'$\tau_{c}$', fontsize=25)

def propensity_2d(model, dataset, ax):
    X = dataset.Xeq_n[dataset.test_idx]
    Y = dataset.Y[dataset.test_idx]
    xmin, xmax = X[:,0].min()-1, X[:,0].max()+1
    ymin, ymax = X[:,1].min()-1, X[:,1].max()+1
    cx = np.linspace(xmin, xmax, 100)
    cy = np.linspace(ymin, ymax, 100)
    CX, CY = np.meshgrid(cx, cy)
    Z = np.zeros(CX.shape)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i,j] = model.e(dataset.etransform(np.array([[CX[i,j], CY[i,j]]])))  
    ax.contourf(CX, CY, Z, levels=100, cmap="bwr", alpha=0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(r'$P_{c, eq}$', fontsize=25)
    ax.set_ylabel(r'$\tau_{c, eq}$', fontsize=25)

def model_2d(model, dataset, ax):
    X = dataset.Xeq_n[dataset.test_idx]
    Y = dataset.Y[dataset.test_idx]
    xmin, xmax = X[:,0].min()-1, X[:,0].max()+1
    ymin, ymax = X[:,1].min()-1, X[:,1].max()+1
    cx = np.linspace(xmin, xmax, 100)
    cy = np.linspace(ymin, ymax, 100)
    CX, CY = np.meshgrid(cx, cy)
    Z = np.zeros(CX.shape)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i,j] = model.e(dataset.etransform(np.array([[CX[i,j], CY[i,j]]]))) * model.eta(dataset.ctransform(np.array([[CX[i,j], CY[i,j]]])))
    ax.contourf(CX, CY, Z, levels=100, cmap="RdBu_r", alpha=0.3)


def plot_train_dataset(dataset, i1, i2, real_classes=False):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
    scatter(dataset, i1, i2, axes[0], 'dim', 'train', real_classes)
    scatter(dataset, i1, i2, axes[1], 'pred', 'train')
    axes[0].legend()
    axes[1].legend()
    plt.show()

def plot_results(model, dataset, real_classes=False, filename='fig.png'):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(17,7))    
    scatter(dataset, 0, 1, axes[0], 'dim', 'test', real_classes)
    scatter(dataset, 0, 1, axes[1], 'pred', 'test')
    # scatter(dataset, 0, 1, axes[2], 'pred', 'test')
    threshold = sum(model.expectation(dataset.X_nt[dataset.train_idx], dataset.Xeq_nt[dataset.train_idx], dataset.Y[dataset.train_idx]))/len(dataset.train_idx)
    criterion_2d(model, dataset, axes[0], threshold=threshold, real_classes=real_classes)
    propensity_2d(model, dataset, axes[1])
    # model_2d(model, dataset, axes[2])
    axes[0].set_title(r'Fatigue criterion ($\eta$)', fontsize=25)
    axes[1].set_title(r'Propensity ($e$)', fontsize=25)
    # axes[2].set_title('At f Fn : probability of crack initiation', fontsize=15)
    axes[0].legend(fontsize=20)
    axes[1].legend(fontsize=20)
    plt.savefig(filename, bbox_inches='tight')