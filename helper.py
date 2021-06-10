from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib import pyplot
import pandas as pd
import seaborn as sn


def plot_cm(y_true, y_pred, figsize=(5, 5)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    print(cm)

    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
            cm[i, j] = p

    sn.set(font_scale=1.4)
    cm = pd.DataFrame(cm, index=['Good', 'Fail'], columns=['Good', 'Fail'])
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = pyplot.subplots(figsize=figsize)
    sn.heatmap(cm, cmap="YlGnBu", annot=annot, fmt='', ax=ax, vmax=100, vmin=0)