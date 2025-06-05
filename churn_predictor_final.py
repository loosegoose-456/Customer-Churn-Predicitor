#✅ XGBoost with RandomizedSearchCV
#✅ Ensemble Voting Classifier (XGBoost + Random Forest + Gradient Boosting)
#✅ Threshold tuning for improved recall
#✅ Advanced feature engineering
#✅ SMOTE, Scaling, and Label Encoding
#✅ Confusion Matrix Heatmap + Plotly 3D scatter
#✅ Top feature importance display

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# 📥 Load dataset
df = pd.read_csv('Telco-Customer-Churn.csv')
df.drop('customerID', axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# 🧠 Advanced Feature Engineering
df['TotalCost'] = df['MonthlyCharges'] * df['tenure']
df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
df['HasMultipleServices'] = (df[['PhoneService','OnlineSecurity','OnlineBackup',
                                 'DeviceProtection','TechSupport','StreamingTV',
                                 'StreamingMovies']] == 'Yes').sum(axis=1)
df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], labels=[1, 2, 3, 4])

# 🔁 Encode categorical variables
for col in df.select_dtypes(include='object').columns:
    if col != 'Churn':
        df[col] = LabelEncoder().fit_transform(df[col])
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
df['TenureGroup'] = df['TenureGroup'].astype(int)

# 🎯 Define features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# 📊 Scale and balance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# 📤 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 🔍 Randomized Search for Best XGBoost
param_grid = {
    'n_estimators': [300, 500],
    'max_depth': [4, 5, 6],
    'learning_rate': [0.01, 0.02, 0.05],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.9],
    'min_child_weight': [1, 5],
    'gamma': [0, 0.1, 0.3]
}

search = RandomizedSearchCV(
    estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    param_distributions=param_grid,
    n_iter=50,
    scoring='accuracy',
    n_jobs=-1,
    cv=5,
    verbose=1,
    random_state=42
)

search.fit(X_train, y_train)
best_xgb = search.best_estimator_

# 🤖 Build ensemble
ensemble = VotingClassifier(estimators=[
    ('xgb', best_xgb),
    ('rf', RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, random_state=42))
], voting='soft')

ensemble.fit(X_train, y_train)

# 🎯 Threshold Tuning
y_proba = ensemble.predict_proba(X_test)[:, 1]
threshold = 0.45
y_pred = (y_proba >= threshold).astype(int)

# 📋 Evaluation
print("\n✅ Classification Report (Threshold = 0.45):")
print(classification_report(y_test, y_pred))

# 🔥 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Voting Ensemble + Tuned Threshold)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 📈 3D Plotly Scatter
df_viz = df.copy()
df_viz['ChurnLabel'] = df_viz['Churn'].map({0: 'No', 1: 'Yes'})
fig = px.scatter_3d(df_viz,
                    x='MonthlyCharges',
                    y='tenure',
                    z='TotalCharges',
                    color='ChurnLabel',
                    title='3D View: Churn by MonthlyCharges, Tenure, TotalCharges',
                    opacity=0.6)
fig.show()

# 📌 Top Feature Importances from XGBoost
importances = best_xgb.feature_importances_
features = X.columns
sorted_idx = np.argsort(importances)[::-1]

print("\n🔎 Top 10 Features by XGBoost Importance:")
for i in sorted_idx[:10]:
    print(f"{features[i]}: {importances[i]:.4f}")
