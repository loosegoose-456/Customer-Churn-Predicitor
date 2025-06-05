#Random Forest Classifier 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load and prepare dataset
df = pd.read_csv('Telco-Customer-Churn.csv')
df.drop('customerID', axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# 🔧 Feature Engineering
df['TotalCost'] = df['MonthlyCharges'] * df['tenure']
df['HasMultipleServices'] = (df[['PhoneService','OnlineSecurity','OnlineBackup',
                                 'DeviceProtection','TechSupport','StreamingTV',
                                 'StreamingMovies']] == 'Yes').sum(axis=1)

# Encode categorical features
for col in df.select_dtypes(include='object').columns:
    if col != 'Churn':
        df[col] = LabelEncoder().fit_transform(df[col])

df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🧪 SMOTE to balance dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 🔍 GridSearchCV for Random Forest optimization
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [8, 10, 12],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Evaluate best model
y_pred = best_model.predict(X_test)

print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 🎯 Feature Importance
importances = best_model.feature_importances_
features = X.columns
sorted_idx = np.argsort(importances)[::-1]

print("\n🔎 Top Features by Importance:")
for i in sorted_idx:
    print(f"{features[i]}: {importances[i]:.4f}")
