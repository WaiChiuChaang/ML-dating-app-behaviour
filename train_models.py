"""Train and evaluate three models on the dating app dataset.

Models included:
- Random Forest (supervised)
- Support Vector Machine (supervised)
- K-Means (unsupervised clustering)

This script mirrors preprocessing done in the notebook and prints evaluation metrics.
"""
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

RANDOM_STATE = 42
DATA_PATH = 'dating_app_behavior_dataset_extended1.csv'


def load_and_preprocess(path=DATA_PATH):
    df_raw = pd.read_csv(path)
    df = df_raw.copy()

    # Drop known redundant columns if present
    for c in ['app_usage_time_label', 'swipe_right_label']:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # Binary target
    positive_outcomes = {'Mutual Match', 'Instant Match', 'Date Happened', 'Relationship Formed'}
    df['target'] = df['match_outcome'].apply(lambda x: 1 if x in positive_outcomes else 0)
    if 'match_outcome' in df.columns:
        df.drop(columns=['match_outcome'], inplace=True)

    # income_bracket map
    if 'income_bracket' in df.columns:
        income_map = {
            'Very Low': 'Low', 'Low': 'Low',
            'Lower-Middle': 'Middle', 'Middle': 'Middle', 'Upper-Middle': 'Middle',
            'High': 'High', 'Very High': 'High'
        }
        df['income_bracket'] = df['income_bracket'].map(income_map)
        df['income_enc'] = OrdinalEncoder(categories=[['Low', 'Middle', 'High']]).fit_transform(df[['income_bracket']])
        df.drop(columns=['income_bracket'], inplace=True)

    # education_level map
    if 'education_level' in df.columns:
        def map_education(val):
            val = str(val)
            if any(k in val for k in ['No Formal', 'High School', 'Diploma']):
                return 'Low'
            elif any(k in val for k in ['Associate', 'Bachelor']):
                return 'Middle'
            elif any(k in val for k in ['Master', 'MBA', 'PhD', 'Postdoc']):
                return 'High'
            return 'Low'
        df['education_level'] = df['education_level'].apply(map_education)
        df['education_enc'] = OrdinalEncoder(categories=[['Low', 'Middle', 'High']]).fit_transform(df[['education_level']])
        df.drop(columns=['education_level'], inplace=True)

    # One-hot nominal columns if present
    nominal_cols = [
        'gender', 'sexual_orientation', 'location_type', 'swipe_time_of_day',
        'body_type', 'relationship_intent', 'zodiac_sign'
    ]
    present_nominal = [c for c in nominal_cols if c in df.columns]
    if present_nominal:
        df = pd.get_dummies(df, columns=present_nominal, drop_first=False, dtype=int)

    # Multi-hot interest tags
    if 'interest_tags' in df.columns:
        mlb = MultiLabelBinarizer()
        interests_split = df['interest_tags'].str.split(', ')
        interest_dummies = pd.DataFrame(
            mlb.fit_transform(interests_split),
            columns=['interest_' + c for c in mlb.classes_],
            index=df.index
        )
        df = pd.concat([df, interest_dummies], axis=1)
        df.drop(columns=['interest_tags'], inplace=True)

    # Numeric scaling
    numeric_cols = [
        'age', 'height_cm', 'weight_kg', 'app_usage_time_min', 'swipe_right_ratio',
        'likes_received', 'mutual_matches', 'profile_pics_count', 'bio_length',
        'message_sent_count', 'emoji_usage_rate', 'last_active_hour'
    ]
    present_numeric = [c for c in numeric_cols if c in df.columns]
    if present_numeric:
        scaler = StandardScaler()
        df[present_numeric] = scaler.fit_transform(df[present_numeric])

    # Feature matrix and target
    X = df.drop(columns=['target'])
    y = df['target']

    # Feature selection: union of top-40 by F-score and MI (like notebook)
    selector_f = SelectKBest(score_func=f_classif, k='all')
    selector_f.fit(X, y)
    f_scores = pd.DataFrame({'feature': X.columns, 'f_score': selector_f.scores_}).sort_values('f_score', ascending=False)

    mi_scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    mi_df = pd.DataFrame({'feature': X.columns, 'mi_score': mi_scores}).sort_values('mi_score', ascending=False)

    top_f = set(f_scores.head(40)['feature'])
    top_mi = set(mi_df.head(40)['feature'])
    selected = sorted(top_f.union(top_mi))

    X_selected = X[selected]

    return X_selected, y


def train_supervised(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)

    svc = LinearSVC(class_weight='balanced', max_iter=20000, dual=False, random_state=RANDOM_STATE)
    svc.fit(X_train, y_train)

    return rf, svc


def evaluate_supervised(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)

        # Try probability estimates, otherwise use decision_function for ROC-AUC
        y_prob = None
        if hasattr(model, 'predict_proba'):
            try:
                y_prob = model.predict_proba(X_test)[:, 1]
            except Exception:
                y_prob = None
        if y_prob is None and hasattr(model, 'decision_function'):
            try:
                y_prob = model.decision_function(X_test)
            except Exception:
                y_prob = None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan

        results[name] = {
            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': roc,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }

    return results


def evaluate_kmeans(X, y, n_clusters=2):
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X)

    ari = adjusted_rand_score(y, labels)
    nmi = normalized_mutual_info_score(y, labels)
    sil = silhouette_score(X, labels)

    # Map cluster -> positive rate
    df = pd.DataFrame({'cluster': labels, 'target': y.values})
    cluster_rates = df.groupby('cluster')['target'].mean().to_dict()

    return {'ari': ari, 'nmi': nmi, 'silhouette': sil, 'cluster_positive_rate': cluster_rates}


def main():
    print('Loading and preprocessing dataset...')
    X, y = load_and_preprocess()

    print(f'Final feature matrix: {X.shape}, target: {y.shape}')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    print('\nTraining supervised models (Random Forest, SVM)...')
    rf, svc = train_supervised(X_train, y_train)

    print('\nEvaluating supervised models...')
    results = evaluate_supervised({'RandomForest': rf, 'SVM': svc}, X_test, y_test)
    for name, res in results.items():
        print(f'\n=== {name} ===')
        print(f"Accuracy: {res['accuracy']:.4f}  Precision: {res['precision']:.4f}  Recall: {res['recall']:.4f}  F1: {res['f1']:.4f}  ROC-AUC: {res['roc_auc']:.4f}")
        print('Confusion matrix:\n', res['confusion_matrix'])

    print('\nPerforming K-Means clustering (unsupervised) on selected features...')
    km_res = evaluate_kmeans(X, y, n_clusters=2)
    print('\nK-Means evaluation:')
    print(f"Adjusted Rand Index: {km_res['ari']:.4f}")
    print(f"Normalized Mutual Info: {km_res['nmi']:.4f}")
    print(f"Silhouette Score: {km_res['silhouette']:.4f}")
    print('Cluster -> positive_rate:', km_res['cluster_positive_rate'])


if __name__ == '__main__':
    main()
