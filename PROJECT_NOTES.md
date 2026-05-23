# 💘 Tying the Data Knot: Predicting Meaningful Connections
### WIA1006/WID3006 Machine Learning — Group Assignment Documentation
**Sem 2, Session 2025/2026 | FCSIT, Universiti Malaya**

---

## 📌 Project Overview

**Project Name:** Tying the Data Knot: Predicting Meaningful Connections

**Objective:** Predict whether a dating app user will achieve a **meaningful connection** based on their demographic profile and in-app behaviour patterns.

**ML Task Type:** Binary Classification
- **Positive (1):** Mutual Match, Instant Match, Date Happened, Relationship Formed
- **Negative (0):** Ghosted, Blocked, Catfished, Chat Ignored, No Action, One-sided Like

**Dataset:** `dating_app_behavior_dataset_extended1.csv`
- 50,000 records × 25 features
- Zero missing values, zero duplicates
- Balanced multi-class target (~5,000 per class), ~40/60 binary split

---

## 📁 Repository Files

| File | Description |
|---|---|
| `ML_dating_app_behaviour.ipynb` | Main Jupyter notebook — full pipeline (105 cells) |
| `dating_app_behavior_dataset.csv` | Original dataset (50k × 19 features, 7.6 MB) |
| `dating_app_behavior_dataset_extended1.csv` | **Extended dataset used** (50k × 25 features, 9.6 MB) |
| `PROJECT_NOTES.md` | This documentation file |

---

## 🆕 Why We Use the Extended Dataset

The extended dataset adds **6 new features** not present in the original:

| New Feature | Type | Why It Matters |
|---|---|---|
| `age` | Numeric (18–59) | Core dating preference factor |
| `height_cm` | Numeric (145–200) | Physical profile signal |
| `weight_kg` | Numeric | Physical profile signal |
| `body_type` | Categorical (6 types) | Profile completeness & preference signal |
| `relationship_intent` | Categorical (6 types) | **Strong predictor** — e.g. Serious vs Hookups |
| `zodiac_sign` | Categorical (12 signs) | Cultural/personality correlation |

---

## 🗂️ Dataset — Feature Breakdown

### Categorical Features

| Feature | Type | Unique Values | Encoding Used |
|---|---|---|---|
| `gender` | Nominal | Female, Male, Non-binary, Transgender, Genderfluid, Prefer Not to Say (6) | One-Hot |
| `sexual_orientation` | Nominal | Straight, Gay, Lesbian, Bisexual, Pansexual, Asexual, Queer, Demisexual (8) | One-Hot |
| `location_type` | Nominal | Urban, Suburban, Rural, Small Town, Remote Area, Metro (6) | One-Hot |
| `income_bracket` | **Ordinal** | 7 levels → consolidated to Low / Middle / High | Ordinal (0/1/2) |
| `education_level` | **Ordinal** | 9 levels → consolidated to Low / Middle / High | Ordinal (0/1/2) |
| `interest_tags` | Multi-value | 49 unique tags (3 per user, comma-separated) | Multi-Hot (49 binary cols) |
| `body_type` | Nominal | Slim, Curvy, Average, Athletic, Muscular, Plus Size (6) | One-Hot |
| `relationship_intent` | Nominal | Serious Relationship, Casual Dating, Hookups, Friends Only, Exploring, Networking (6) | One-Hot |
| `zodiac_sign` | Nominal | 12 signs | One-Hot |
| `swipe_time_of_day` | Nominal | Morning, Afternoon, Evening, Late Night, After Midnight, Early Morning (6) | One-Hot |
| `app_usage_time_label` | Redundant | String version of `app_usage_time_min` | **Dropped** |
| `swipe_right_label` | Redundant | String version of `swipe_right_ratio` | **Dropped** |
| `match_outcome` | **Target** | 10 classes | Binarised → `target` |

### Numerical Features

| Feature | Range | Description | Normalization |
|---|---|---|---|
| `age` | 18–59 | User age | StandardScaler |
| `height_cm` | 145–200 | Height in cm | StandardScaler |
| `weight_kg` | varies | Weight in kg | StandardScaler |
| `app_usage_time_min` | varies | Daily app usage in minutes | StandardScaler |
| `swipe_right_ratio` | 0.0–1.0 | Ratio of right swipes (already 0–1) | StandardScaler |
| `likes_received` | varies | Number of likes received | StandardScaler |
| `mutual_matches` | 0–30 | Number of mutual matches | StandardScaler |
| `profile_pics_count` | 0–6 | Number of profile photos | StandardScaler |
| `bio_length` | varies | Character count of bio | StandardScaler |
| `message_sent_count` | varies | Total messages sent | StandardScaler |
| `emoji_usage_rate` | 0.0–0.94 | Proportion of messages with emojis (already 0–1) | StandardScaler |
| `last_active_hour` | 0–23 | Hour of day most active | StandardScaler |

---

## ⚙️ Pipeline — Step by Step

### Step 1: Data Loading
```python
df_raw = pd.read_csv('dating_app_behavior_dataset_extended1.csv')
# Shape: (50000, 25)
```
- Loaded from CSV (local) or Google Drive (Colab)
- No preprocessing at this stage — `df_raw` is kept untouched for EDA

---

### Step 2: Exploratory Data Analysis (EDA)

**What was explored:**

1. **Basic Info** — `df.info()`, `df.describe()` for all 25 columns
2. **Missing values** — None found
3. **Duplicates** — None found
4. **Target distribution** — All 10 `match_outcome` classes are balanced (~5,000 each); binary split is 39.7% Positive / 60.3% Negative
5. **Categorical distributions** — Bar charts for all 9 categorical columns
6. **Numerical distributions** — Histograms for all 12 numeric columns
7. **Outlier detection** — Boxplots for all 12 numeric columns; no extreme outliers found
8. **Feature vs Target** — Overlaid histograms (numeric) and stacked % bars (categorical) split by Positive/Negative outcome
9. **Correlation heatmap** — Pearson correlation among all 12 numeric features
10. **Interest tag analysis** — Frequency chart of all 49 unique interest tags

**Key EDA findings:**
- All features are uniformly distributed — the dataset is synthetically generated and well-balanced
- No strong linear correlations between numeric features (expected for synthetic data)
- `relationship_intent` and some interest tags show slight variation in positive match rates
- No outlier removal needed — all numeric ranges are plausible

---

### Step 3: Data Preprocessing

#### 3.1 Drop Redundant Columns
```python
df.drop(columns=['app_usage_time_label', 'swipe_right_label'], inplace=True)
```
- `app_usage_time_label` is just a string category of `app_usage_time_min` → redundant
- `swipe_right_label` is just a string category of `swipe_right_ratio` → redundant

#### 3.2 Create Binary Target
```python
positive_outcomes = {'Mutual Match', 'Instant Match', 'Date Happened', 'Relationship Formed'}
df['target'] = df['match_outcome'].apply(lambda x: 1 if x in positive_outcomes else 0)
df.drop(columns=['match_outcome'], inplace=True)
```
- **Why binary?** Higher accuracy, clearer metrics (ROC-AUC, F1), easier to present
- **Positive class:** 4 outcomes representing actual human connection
- **Negative class:** 6 outcomes representing failed/one-sided interactions
- **Result:** 19,850 Positive (39.7%) / 30,150 Negative (60.3%)

#### 3.3 Ordinal Encoding — income_bracket
```python
income_map = {
    'Very Low': 'Low',   'Low': 'Low',
    'Lower-Middle': 'Middle', 'Middle': 'Middle', 'Upper-Middle': 'Middle',
    'High': 'High',      'Very High': 'High'
}
# Then: OrdinalEncoder(categories=[['Low', 'Middle', 'High']])
```
- **Why ordinal?** Income has a natural order (Low < Middle < High)
- Consolidates 7 granular levels → 3 meaningful tiers → encoded as 0, 1, 2

#### 3.4 Ordinal Encoding — education_level
```python
def map_education(val):
    if any(k in val for k in ['No Formal', 'High School', 'Diploma']): return 'Low'
    elif any(k in val for k in ['Associate', 'Bachelor']): return 'Middle'
    elif any(k in val for k in ['Master', 'MBA', 'PhD', 'Postdoc']): return 'High'
```
- **Why keyword matching?** The CSV uses curly apostrophes (`Bachelor's` → `Bachelor\u2019s`) which break exact string matching
- Consolidates 9 qualification levels → 3 academic tiers → encoded as 0, 1, 2

#### 3.5 One-Hot Encoding — Nominal Categoricals
```python
nominal_cols = ['gender', 'sexual_orientation', 'location_type',
                'swipe_time_of_day', 'body_type', 'relationship_intent', 'zodiac_sign']
df = pd.get_dummies(df, columns=nominal_cols, drop_first=False, dtype=int)
```
- **Why one-hot?** These features have no natural order — all values are equally valid
- `drop_first=False` keeps all categories (avoids dummy variable trap only an issue for linear regression, which is not our primary model)
- Results in ~43 new binary columns

#### 3.6 Multi-Hot Encoding — interest_tags
```python
mlb = MultiLabelBinarizer()
interests_split = df['interest_tags'].str.split(', ')
interest_dummies = pd.DataFrame(mlb.fit_transform(interests_split),
                                columns=['interest_' + c for c in mlb.classes_])
```
- Each user has exactly 3 interests stored as a comma-separated string
- `MultiLabelBinarizer` creates 1 binary column per unique tag
- Results in **49 new binary columns** (one per unique interest)

#### 3.7 StandardScaler Normalization
```python
numeric_cols = ['age', 'height_cm', 'weight_kg', 'app_usage_time_min',
                'swipe_right_ratio', 'likes_received', 'mutual_matches',
                'profile_pics_count', 'bio_length', 'message_sent_count',
                'emoji_usage_rate', 'last_active_hour']
df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])
```
- **Why StandardScaler?** Transforms each column to mean=0, std=1
- Essential for distance-based models (KNN, SVM) and gradient descent models (Logistic Regression)
- Tree-based models (Random Forest, XGBoost) don't require normalization but it doesn't hurt them

**After all preprocessing: shape = (50000, 114)** — 113 feature columns + 1 target column

---

### Step 4: Feature Selection

**Goal:** Reduce from 113 features to a smaller set of the most informative features, reducing noise and training time.

#### Method 1: ANOVA F-Score (SelectKBest)
```python
selector_f = SelectKBest(score_func=f_classif, k='all')
selector_f.fit(X, y)
```
- Tests whether the mean of each feature differs significantly between Positive and Negative classes
- Higher F-score = more statistically significant difference between classes
- All features scored; we rank them and take the **top 40**

#### Method 2: Mutual Information
```python
mi_scores = mutual_info_classif(X, y, random_state=42)
```
- Measures how much information each feature provides about the target
- Non-linear — captures relationships that F-score misses
- More robust for binary-encoded features (interest tags, one-hot columns)
- All features scored; take the **top 40**

#### Final Selection
```python
selected_features = sorted(set(f_scores.head(40)['feature']).union(set(mi_df.head(40)['feature'])))
# Result: 67 features selected
```
- **Union strategy:** Keep any feature that ranks highly in either method
- **Result: 67 features** selected from 113 total

> **Note:** Top-scoring features include numeric columns (`mutual_matches`, `likes_received`, `message_sent_count`) and relationship_intent one-hot columns, suggesting these are the strongest predictors.

---

### Step 5: PCA (Dimensionality Reduction)

**Goal:** Optionally reduce 67 features further by projecting into principal component space. Used as an alternative feature set to compare model performance.

```python
pca_full = PCA(random_state=42)
pca_full.fit(X_selected)
cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100
n_components_95 = int(np.argmax(cumvar >= 95) + 1)  # = 55 components
```

**What PCA does:**
1. Computes the directions of maximum variance in the feature space (principal components)
2. Projects the data onto these directions
3. The first component captures the most variance, each subsequent component less

**Results:**
- **55 components** retain **95.2% of total variance**
- Reduces from 67 → 55 dimensions
- Components are linear combinations of original features — **interpretability is lost**

**Two feature sets maintained for modeling:**
| Variable | Shape | Description |
|---|---|---|
| `X_selected` | (50000, 67) | Original 67 selected features |
| `X_pca` | (50000, 55) | PCA-reduced to 55 components |

---

### Step 6: Train / Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y,
    test_size=0.2,
    random_state=42,
    stratify=y       # ensures same class ratio in both splits
)
```

**Why stratify?**
Without `stratify=y`, a random split might put more Positive examples in train than test, making evaluation unreliable. Stratification ensures both splits have the same ~39.7% / 60.3% ratio.

**Result:**

| Split | Rows | Positive | Negative |
|---|---|---|---|
| Training | 40,000 | ~15,880 (39.7%) | ~24,120 (60.3%) |
| Test | 10,000 | ~3,970 (39.7%) | ~6,030 (60.3%) |

---

### Step 7: Model Training (Section 9 in notebook)

We train **6 different models** to compare their performance on the same data:

| # | Model | Type | Why Selected | Key Parameters |
|---|---|---|---|---|
| 1 | **Logistic Regression** | Linear | Baseline, interpretable, fast | `max_iter=1000, solver='lbfgs'` |
| 2 | **K-Nearest Neighbors** | Instance-based | Distance-based, non-parametric | `n_neighbors=5` |
| 3 | **Decision Tree** | Tree-based | Fully interpretable | default |
| 4 | **Random Forest** | Ensemble (Bagging) | Robust, handles high dims | `n_estimators=200` |
| 5 | **XGBoost** | Ensemble (Boosting) | Usually best on tabular data | `n_estimators=200, eval_metric='logloss'` |
| 6 | **SVM** | Kernel-based | Good with clear margins | `kernel='rbf', probability=True` |

#### How each model works:

**Logistic Regression:**
Fits a linear decision boundary by learning weights for each feature. Output is a probability via the sigmoid function. Simple and interpretable — serves as a baseline to beat.

**K-Nearest Neighbors (KNN):**
Classifies a point by looking at its K nearest neighbours in feature space. No training phase — all computation happens at prediction time. Sensitive to feature scaling (which is why we StandardScaled).

**Decision Tree:**
Recursively splits the data on the feature that best separates the classes (using Gini impurity or entropy). Very interpretable but prone to overfitting if not pruned.

**Random Forest:**
Trains many decision trees on random subsets of the data and features, then aggregates their predictions (majority vote). Reduces overfitting compared to single trees. Provides feature importance scores.

**XGBoost (Extreme Gradient Boosting):**
Builds trees sequentially — each new tree corrects the errors of the previous ones. Uses gradient descent to minimize the loss function. Typically the strongest performer on structured/tabular data.

**Support Vector Machine (SVM):**
Finds the hyperplane that maximally separates the two classes. The `rbf` kernel maps data into a higher-dimensional space where linear separation is possible. `probability=True` enables probability estimates via Platt scaling.

#### What we record for each model:
- **Training accuracy** — how well it fits the training data
- **Test accuracy** — how well it generalises to unseen data
- **Precision** — of predicted positives, how many are actually positive
- **Recall** — of actual positives, how many were correctly predicted
- **F1 Score** — harmonic mean of precision and recall (balances both)
- **ROC-AUC** — area under the ROC curve (1.0 = perfect, 0.5 = random)
- **Training time** — wall clock time in seconds
- **Overfitting gap** — (train accuracy - test accuracy); large gap = overfitting

#### Evaluation visualisations produced:
1. **Model comparison bar chart** — side-by-side comparison of accuracy, precision, recall, F1, ROC-AUC
2. **Confusion matrices** — 2×2 heatmaps for each model (True/False × Positive/Negative)
3. **ROC curves** — all 6 models overlaid on one plot with AUC values
4. **Classification reports** — per-class precision, recall, F1 for each model
5. **5-fold cross-validation boxplot** — shows stability of each model across different data splits
6. **Learning curves** — training vs validation accuracy as training set size grows (top 3 models)

---

### Step 8: Hyperparameter Tuning (Section 10 in notebook)

**Goal:** Improve the top 3 models by searching for better hyperparameter combinations.

**Method:** `RandomizedSearchCV` with:
- **30 random parameter combinations** per model
- **5-fold cross-validation** per combination
- **F1 score** as the optimisation metric (balances precision and recall)
- Total: 30 × 5 = 150 model fits per model being tuned

#### Parameter search spaces:

**Random Forest:**
| Parameter | Values Searched |
|---|---|
| `n_estimators` | 100, 200, 300, 500 |
| `max_depth` | None, 10, 20, 30, 50 |
| `min_samples_split` | 2, 5, 10 |
| `min_samples_leaf` | 1, 2, 4 |
| `max_features` | 'sqrt', 'log2', None |

**XGBoost:**
| Parameter | Values Searched |
|---|---|
| `n_estimators` | 100, 200, 300, 500 |
| `max_depth` | 3, 5, 7, 10 |
| `learning_rate` | 0.01, 0.05, 0.1, 0.2 |
| `subsample` | 0.6, 0.8, 1.0 |
| `colsample_bytree` | 0.6, 0.8, 1.0 |
| `min_child_weight` | 1, 3, 5 |

**SVM:**
| Parameter | Values Searched |
|---|---|
| `C` | 0.1, 1, 10, 100 |
| `gamma` | 'scale', 'auto', 0.01, 0.001 |
| `kernel` | 'rbf', 'poly' |

**KNN:**
| Parameter | Values Searched |
|---|---|
| `n_neighbors` | 3, 5, 7, 11, 15, 21 |
| `weights` | 'uniform', 'distance' |
| `metric` | 'euclidean', 'manhattan', 'minkowski' |

**Decision Tree:**
| Parameter | Values Searched |
|---|---|
| `max_depth` | None, 5, 10, 20, 30 |
| `min_samples_split` | 2, 5, 10, 20 |
| `min_samples_leaf` | 1, 2, 4, 8 |
| `criterion` | 'gini', 'entropy' |

**Logistic Regression:**
| Parameter | Values Searched |
|---|---|
| `C` | 0.01, 0.1, 1, 10, 100 |
| `penalty` | 'l2' |
| `solver` | 'lbfgs', 'liblinear' |

#### Tuning output:
- **Best parameters** found for each model
- **Before vs After comparison** — shows accuracy, F1, ROC-AUC change
- **Bar chart** comparing baseline vs tuned for top 3 models
- **Best overall model** selected by highest F1 score
- **Confusion matrix and ROC curve** for the best tuned model

---

### Step 9: Feature Importance Analysis (Section 11 in notebook)

Uses the best tree-based model (Random Forest or XGBoost) to extract feature importance scores:
```python
feat_imp = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importance_model.feature_importances_
}).sort_values('importance', ascending=False)
```

**What feature importance means:**
- For Random Forest: average decrease in impurity (Gini) when splitting on that feature
- For XGBoost: total gain from splits on that feature across all trees
- Higher importance = that feature is more useful for distinguishing Positive from Negative outcomes

**Output:** Top 20 features ranked by importance, with horizontal bar chart.

---

### Step 10: Final Summary (Section 12 in notebook)

- **Comprehensive comparison table** of all baseline + tuned models, sorted by F1 score
- **Final bar chart** ranking all models (green = tuned, grey = baseline)
- **Best model selection** with detailed classification report

---

## 📊 Full Pipeline Diagram

```
dating_app_behavior_dataset_extended1.csv  (50,000 x 25)
        |
        v
  [EDA]  ->  visualizations, distributions, correlations
        |
        v
  [Drop] app_usage_time_label, swipe_right_label  ->  50,000 x 23
        |
        v
  [Binary Target]  match_outcome  ->  target (0/1)
        |
        v
  [Ordinal Encode]  income, education  ->  income_enc, education_enc
        |
        v
  [One-Hot Encode]  7 nominal columns  ->  +43 binary columns
        |
        v
  [Multi-Hot Encode]  interest_tags (49 tags)  ->  +49 binary columns
        |
        v
  [StandardScaler]  12 numeric columns  ->  mean=0, std=1
        |
        v
  Feature matrix X: (50,000 x 113)
        |
        |--[ANOVA F top 40]---+
        |                     +--[Union]--> X_selected (50,000 x 67)
        +--[MI top 40]--------+                  |
                                                 |--[PCA 95%]--> X_pca (50,000 x 55)
                                                 |
                                                 v
                                    [Train/Test Split 80/20 stratified]
                                                 |
                               +-----------------+------------------+
                               v                                    v
                        X_train (40k x 67)                  X_test (10k x 67)
                               |
                               v
                    [Train 6 Baseline Models]
                    1. Logistic Regression
                    2. KNN
                    3. Decision Tree
                    4. Random Forest
                    5. XGBoost
                    6. SVM
                               |
                               v
                    [Evaluate: Acc, F1, ROC-AUC, Confusion Matrix]
                    [5-Fold Cross-Validation]
                    [Learning Curves for Top 3]
                               |
                               v
                    [Hyperparameter Tuning - Top 3 Models]
                    RandomizedSearchCV (30 iter, 5-fold CV)
                               |
                               v
                    [Final Comparison: Baseline vs Tuned]
                    [Best Model Selection by F1 Score]
                               |
                               v
                    [Feature Importance Analysis]
```

---

## 🛠️ Technical Notes

### Running Locally (Default Setup)
1. Ensure the CSV files (e.g., `dating_app_behavior_dataset_extended1.csv`) are in the same directory as the notebook.
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn`
3. The local path is now configured directly in Section 2 (`DATA_PATH = 'dating_app_behavior_dataset_extended1.csv'`).

### Running in Google Colab
1. Upload the CSV files to your Google Drive under `MyDrive/Dataset/` (or upload directly to the Colab session files).
2. If using Google Drive, mount Drive in Colab and change `DATA_PATH` in Section 2 to your Drive path:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   DATA_PATH = '/content/drive/MyDrive/Dataset/dating_app_behavior_dataset_extended1.csv'
   ```
3. Run all cells top to bottom — Section 1 installs all required packages.

### Important: Education Level Encoding
The CSV stores values like `Bachelor's` with a curly apostrophe (`\u2019`), not a straight apostrophe (`'`). Direct dictionary mapping with `str.map()` would leave these as `NaN`. We use keyword-based matching instead:
```python
def map_education(val):
    val = str(val)
    if any(k in val for k in ['No Formal', 'High School', 'Diploma']): return 'Low'
    elif any(k in val for k in ['Associate', 'Bachelor']): return 'Middle'
    elif any(k in val for k in ['Master', 'MBA', 'PhD', 'Postdoc']): return 'High'
    return 'Low'
```

### RANDOM_STATE = 42
Used in all stochastic operations to ensure full reproducibility:
- `train_test_split`
- `PCA`
- `mutual_info_classif`
- All ML models
- `RandomizedSearchCV`

### XGBoost Fallback
If `xgboost` is not installed, the notebook automatically falls back to sklearn's `GradientBoostingClassifier`:
```python
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
```

---

## 📋 Assignment Submission Checklist

| Requirement | Status |
|---|---|
| Min 5 ML models trained | ✅ Done (6 models) |
| Hyperparameter tuning | ✅ Done (RandomizedSearchCV, top 3 models) |
| Model comparison table | ✅ Done |
| Cross-validation | ✅ Done (5-fold) |
| Learning curves | ✅ Done (top 3 models) |
| Feature importance | ✅ Done |
| Confusion matrices | ✅ Done (all 6 models) |
| ROC curves | ✅ Done (all 6 models overlaid) |
| Auto-sklearn comparison | 🔲 Pending (Colab, Linux) |
| EDA complete | ✅ Done |
| Data preprocessing complete | ✅ Done |
| Feature selection (F-score + MI) | ✅ Done |
| PCA | ✅ Done |
| Train/test split | ✅ Done |
| Presentation slides | 🔲 Pending |
| 5-minute video recording | 🔲 Pending |
| Group project report | 🔲 Pending |
| Submit on SPECTRUM | 🔲 Pending (deadline: 8 June 2026) |

---

## 📓 Notebook Section Index

| Section | Cells | Description |
|---|---|---|
| 1 — Install & Import | 2–4 | Libraries and plot style |
| 2 — Data Loading | 5–7 | Load CSV, column overview |
| 3 — EDA | 8–29 | 10 subsections of exploration and visualisation |
| 4 — Preprocessing | 30–46 | Drop, encode, normalise |
| 5 — Feature Selection | 47–57 | F-Score, MI, union strategy |
| 6 — PCA | 58–64 | Variance analysis, biplot |
| 7 — Train/Test Split | 65–67 | 80/20 stratified |
| 8 — Pre-Training Checklist | 68 | Status summary |
| 9 — Model Training | 69–87 | 6 models, comparison, confusion matrices, ROC, CV, learning curves |
| 10 — Hyperparameter Tuning | 88–99 | RandomizedSearchCV, before/after comparison |
| 11 — Feature Importance | 100–101 | Top 20 features from best tree model |
| 12 — Final Summary | 102–105 | Comprehensive ranking, best model |

---

*Last updated: 23 May 2026*
