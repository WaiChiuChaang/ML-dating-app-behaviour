"""Generate visual outputs for trained models.

Produces and saves:
- Random Forest feature importances
- Confusion matrices for RF and SVM
- ROC and Precision-Recall curves
- PCA 2D scatter with true labels and classifier decision boundary
- K-Means clusters on PCA 2D

Outputs saved to `outputs/` directory.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix

from train_models import load_and_preprocess, train_supervised, evaluate_supervised, evaluate_kmeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

sns.set_theme(style='whitegrid')

OUT_DIR = 'outputs'
os.makedirs(OUT_DIR, exist_ok=True)


def plot_feature_importances(model, feature_names, outpath, top_n=25):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(8, max(4, top_n*0.25)))
    sns.barplot(x=importances[idx], y=np.array(feature_names)[idx], palette='viridis')
    plt.title('Random Forest - Top Feature Importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_confusion_matrix(cm, labels, outpath, title='Confusion Matrix'):
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_roc_pr(y_true, y_score, outprefix, label):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.3f})')
    plt.plot([0,1], [0,1], 'k--', alpha=0.6)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{outprefix}_roc.png')
    plt.close()

    # Precision-Recall
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f'{label} (AP = {ap:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{outprefix}_pr.png')
    plt.close()


def plot_pca_decision_boundary(clf, X, y, outpath, title='PCA decision boundary'):
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)

    # retrain classifier on PCA space
    clf2 = None
    try:
        from sklearn.base import clone
        clf2 = clone(clf)
        clf2.fit(X2, y)
    except Exception:
        clf2 = clf

    x_min, x_max = X2[:,0].min()-1, X2[:,0].max()+1
    y_min, y_max = X2[:,1].min()-1, X2[:,1].max()+1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    try:
        Z = clf2.predict(grid)
    except Exception:
        # if classifier expects original dimensionality, project back (not ideal)
        Z = np.zeros(len(grid))

    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, alpha=0.15, cmap='RdYlBu')
    scatter = plt.scatter(X2[:,0], X2[:,1], c=y, cmap='coolwarm', s=8, alpha=0.7)
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(*scatter.legend_elements(), title='target')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_kmeans_clusters(X, labels_true, n_clusters, outpath):
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = km.fit_predict(X)

    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X2[:,0], y=X2[:,1], hue=clusters, palette='tab10', s=10, legend='full')
    plt.title('K-Means Clusters (PCA-2D)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    X, y = load_and_preprocess()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    rf, svc = train_supervised(X_train, y_train)

    # Random Forest feature importances
    plot_feature_importances(rf, X.columns.tolist(), os.path.join(OUT_DIR, 'rf_feature_importances.png'))

    # Evaluate supervised models and plot ROC/PR and confusion matrices
    res = evaluate_supervised({'RandomForest': rf, 'SVM': svc}, X_test, y_test)
    for name, model in [('RandomForest', rf), ('SVM', svc)]:
        # Confusion matrix
        cm = res[name]['confusion_matrix']
        plot_confusion_matrix(cm, ['Negative','Positive'], os.path.join(OUT_DIR, f'{name}_confusion_matrix.png'), title=f'{name} Confusion Matrix')

        # ROC/PR: obtain decision scores
        y_score = None
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_test)[:,1]
        elif hasattr(model, 'decision_function'):
            y_score = model.decision_function(X_test)
        else:
            y_score = model.predict(X_test)

        plot_roc_pr(y_test, y_score, os.path.join(OUT_DIR, name.lower()), name)

    # PCA decision boundary plot (using entire dataset for visualization)
    plot_pca_decision_boundary(rf, X, y, os.path.join(OUT_DIR, 'rf_pca_decision_boundary.png'), title='RandomForest Decision Boundary (PCA-2D)')

    # K-Means clusters plot
    plot_kmeans_clusters(X, y, n_clusters=2, outpath=os.path.join(OUT_DIR, 'kmeans_clusters.png'))

    print('Plots saved to', OUT_DIR)


if __name__ == '__main__':
    main()
