## How To Load Machine Learning Data

### Considerations When Loading CSV Data

- **What is this?** Preparing to ingest data from raw CSV files.
- **Why do it?** Ensures the data is clean, interpretable, and correctly formatted before modeling.
- **What happens?** You load the data and immediately check its shape, nulls, and types.
- **What can you get?** Early data issues, encoding mismatches, missing values.
- **Task:** Ingest, verify, and prepare data structure.
- **Best Practice:** Always inspect the first few rows and data types.

### Pima Indians Dataset

- **What is this?** A binary classification dataset for diabetes prediction based on diagnostic measurements.
- **Why use it?** It's small, clean, and easy to use for learning ML steps.
- **Task:** Apply classification models to predict diabetes.
- **Example:**

```python
from sklearn.datasets import load_diabetes
import pandas as pd

# Load dataset and convert to DataFrame
X, y = load_diabetes(return_X_y=True, as_frame=True)
df = X.copy()
df['target'] = y
print(df.head())
```

---

### Load CSV Files with the Python Standard Library

- **Why?** Lightweight method without third-party packages.
- **Limitation:** Not ideal for large or complex datasets.

```python
import csv
with open('data.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)
```

### Load CSV Files with NumPy

- **When?** When working with numerical arrays only.

```python
import numpy as np
data = np.loadtxt('data.csv', delimiter=',')
print(data[:5])
```

### Load CSV Files with Pandas

- **Why?** Most flexible, handles types, headers, missing values, etc.

```python
import pandas as pd
df = pd.read_csv('data.csv')
print(df.head())
```

---

## Understand Your Data With Descriptive Statistics

### Peek at Your Data

```python
print(df.head())  # View first 5 rows
```

### Dimensions of Your Data

```python
print(df.shape)  # Rows and columns
```

### Data Type For Each Attribute

```python
print(df.dtypes)  # Check numeric vs object types
```

### Descriptive Statistics

```python
print(df.describe())  # Count, mean, std, etc.
```

### Class Distribution (Classification Only)

```python
print(df['class'].value_counts())  # Useful for imbalance
```

### Correlations Between Attributes

```python
import seaborn as sns
import matplotlib.pyplot as plt

corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.2f')
plt.show()
```

### Skew of Univariate Distributions

```python
print(df.skew())  # For checking non-normality
```

---

## Understand Your Data With Visualization

### Univariate Plots

```python
df.hist(figsize=(10, 10))
plt.tight_layout()
plt.show()
```

### Multivariate Plots

```python
sns.pairplot(df, hue='class')
plt.show()
```

---

## Prepare Your Data For Machine Learning

### Data Transforms

- **Why?** Many ML models perform better with scaled, normalized, or binarized data.

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, Binarizer

# Rescale
rescaled = MinMaxScaler().fit_transform(df)

# Standardize
standardized = StandardScaler().fit_transform(df)

# Normalize
normalized = Normalizer().fit_transform(df)

# Binarize
binary = Binarizer(threshold=0.0).fit_transform(df)
```

---

## Feature Selection For Machine Learning

### Univariate Selection

```python
from sklearn.feature_selection import SelectKBest, chi2
X_new = SelectKBest(score_func=chi2, k=4).fit_transform(X, y)
```

### Recursive Feature Elimination (RFE)

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
rfe = RFE(model, n_features_to_select=3)
rfe.fit(X, y)
print(rfe.support_)
```

### Principal Component Analysis (PCA)

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
```

### Feature Importance

```python
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X, y)
print(model.feature_importances_)
```

---

## Evaluate the Performance of Machine Learning Algorithms with Resampling

### Split into Train and Test Sets

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

### K-fold Cross Validation

```python
from sklearn.model_selection import KFold, cross_val_score
model = LogisticRegression()
kfold = KFold(n_splits=10)
results = cross_val_score(model, X, y, cv=kfold)
```

### Leave One Out Cross Validation

```python
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
results = cross_val_score(model, X, y, cv=loo)
```

### Repeated Random Test-Train Splits

```python
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=10, test_size=0.2)
results = cross_val_score(model, X, y, cv=cv)
```

---

## Machine Learning Algorithm Performance Metrics

### Classification Metrics

```python
from sklearn.metrics import classification_report
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

### Regression Metrics

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print(mean_absolute_error(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))
```

---

## Spot-Check Classification Algorithms

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

models = [
    ('LR', LogisticRegression()),
    ('DT', DecisionTreeClassifier()),
    ('RF', RandomForestClassifier()),
    ('SVM', SVC()),
    ('NB', GaussianNB())
]

for name, model in models:
    scores = cross_val_score(model, X, y, cv=10)
    print(f"{name}: {scores.mean():.3f}")
```

---

## Spot-Check Regression Algorithms

```python
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

models = [
    ('LR', LinearRegression()),
    ('Ridge', Ridge()),
    ('RF', RandomForestRegressor())
]

for name, model in models:
    scores = cross_val_score(model, X, y, cv=10, scoring='r2')
    print(f"{name}: {scores.mean():.3f}")
```

---

## Compare Machine Learning Algorithms

```python
results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=10)
    cv_results = cross_val_score(model, X, y, cv=kfold)
    results.append(cv_results)
    names.append(name)

plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()
```

---

## Automate Machine Learning Workflows with Pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
pipeline.fit(X_train, y_train)
```

---

## Improve Performance with Ensembles

### Bagging

```python
from sklearn.ensemble import BaggingClassifier
model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100)
```

### Boosting

```python
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=100)
```

### Voting Ensemble

```python
from sklearn.ensemble import VotingClassifier
ensemble = VotingClassifier(estimators=models, voting='hard')
```

---

## Improve Performance with Algorithm Tuning

### Grid Search

```python
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid)
grid.fit(X, y)
print(grid.best_params_)
```

### Random Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

param_dist = {'C': uniform(loc=0, scale=4)}
search = RandomizedSearchCV(LogisticRegression(), param_distributions=param_dist, n_iter=100)
search.fit(X, y)
print(search.best_params_)
```

---

## Save and Load Machine Learning Models

### Pickle

```python
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
```

### Joblib

```python
import joblib
joblib.dump(model, 'model.joblib')
```

### Load Model

```python
model = joblib.load('model.joblib')
```

---

## Tips To Remember

- Always inspect raw data before processing.
- Use pipelines for clean, repeatable ML workflows.
- Prefer cross-validation over train/test split for robust evaluation.
- Standardize or normalize data for models sensitive to scale.
- Save your best models and track experiments.
- Watch for data leakage.

---

This is your all-in-one machine learning blueprint. Master these concepts and you’ll never need to reference anything else.

