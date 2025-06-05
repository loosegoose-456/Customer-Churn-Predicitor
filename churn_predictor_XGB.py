#XG Boost
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

# 📌 Load and prepare dataset
df = pd.read_csv('Telco-Customer-Churn.csv')
df.drop('customerID', axis=1, inplace=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# 🎯 Feature Engineering
df['TotalCost'] = df['MonthlyCharges'] * df['tenure']
df['HasMultipleServices'] = (df[['PhoneService','OnlineSecurity','OnlineBackup',
                                 'DeviceProtection','TechSupport','StreamingTV',
                                 'StreamingMovies']] == 'Yes').sum(axis=1)

# 🔁 Encode categorical features
for col in df.select_dtypes(include='object').columns:
    if col != 'Churn':
        df[col] = LabelEncoder().fit_transform(df[col])
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# 🧠 Features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# 🧪 Scale and balance dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# 📊 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 🔍 GridSearchCV to tune XGBoost
param_grid = {
    'n_estimators': [300, 500],              # More trees for better generalization
    'max_depth': [4, 5, 6],                  # Avoid overfitting (not too deep)
    'learning_rate': [0.01, 0.02, 0.05],     # Smaller = slower but better accuracy
    'subsample': [0.7, 0.8],                 # Random row sampling (controls variance)
    'colsample_bytree': [0.7, 0.9],          # Random feature sampling (diversity)
    'min_child_weight': [1, 5],              # Minimum sum of instance weight in a child
    'gamma': [0, 0.1, 0.3]                   # Minimum loss reduction to split
}

search = RandomizedSearchCV(
    estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    param_distributions=param_grid,
    n_iter=50,       # number of combinations to try
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

search.fit(X_train, y_train)
best_model = search.best_estimator_

# 🧾 Make predictions
y_pred = best_model.predict(X_test)

# 📋 Classification Report
print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred))

# 🔥 Heatmap: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('XGBoost Confusion Matrix (Tuned)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 📈 Plotly 3D scatter plot of key features
df_viz = df.copy()
df_viz['ChurnLabel'] = df_viz['Churn'].map({0: 'No', 1: 'Yes'})
fig = px.scatter_3d(df_viz,
                    x='MonthlyCharges',
                    y='tenure',
                    z='TotalCharges',
                    color='ChurnLabel',
                    title='3D Plot: Customer Churn by MonthlyCharges, Tenure, TotalCharges',
                    opacity=0.6)
fig.show()

# 📌 Feature Importance (Top 10)
importances = best_model.feature_importances_
features = X.columns
sorted_idx = np.argsort(importances)[::-1]

print("\n🔎 Top 10 Features by Importance:")
for i in sorted_idx[:10]:
    print(f"{features[i]}: {importances[i]:.4f}")
