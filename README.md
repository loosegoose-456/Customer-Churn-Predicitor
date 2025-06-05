#Customer Churn Prediction Model

This project analyzes customer churn behavior in a telecom company using advanced machine learning models. The goal is to predict which customers are likely to churn based on their usage patterns, services, and account details.

## 📊 Dataset

- **Source**: `Telco-Customer-Churn.csv`
- **Size**: ~7,000 customer records
- **Features**: Demographics, contract details, service usage, billing
- **Target**: `Churn` (Yes/No)

## 🧠 Models Implemented

### 1. `churn_predictor_final.py`  
- ✅ XGBoost with `RandomizedSearchCV`  
- ✅ Ensemble Voting Classifier (XGBoost + Random Forest + Gradient Boosting)  
- ✅ Threshold tuning for improved recall  
- ✅ SMOTE for class balancing  
- ✅ Feature scaling + advanced feature engineering  
- ✅ Confusion matrix heatmap  
- ✅ Interactive 3D Plotly visualization  
- ✅ Top feature importances from XGBoost

### 2. `churn_predictor_RFC.py`  
- 🔁 Random Forest with `GridSearchCV`  
- ⚙️ Feature engineering: `TotalCost`, `HasMultipleServices`  
- 🧪 SMOTE + standard scaling  
- 📋 Classification report and confusion matrix  
- 🎯 Feature importance ranking

### 3. `churn_predictor_XGB.py`  
- 🔍 XGBoost with full `RandomizedSearchCV` tuning  
- 📈 3D Plotly scatter for visualizing churn by key billing features  
- 📌 Top 10 important features shown  
- 🧪 SMOTE + scaler included

### 4. `churn_predictor_TF.py`  
- 🤖 Deep learning with TensorFlow Sequential model  
- 🧱 Dense layers with ReLU + Sigmoid output  
- 📉 Binary cross-entropy loss, Adam optimizer  
- 🔍 Confusion matrix visualization

## 🔧 Feature Engineering Highlights

- `TotalCost`: Estimated total billing based on tenure  
- `AvgMonthlySpend`: Derived average monthly spending  
- `HasMultipleServices`: Count of "Yes" service columns  
- `TenureGroup`: Customer lifecycle segmentation

## 🛠️ Techniques Used

- Label Encoding  
- Standard Scaling  
- SMOTE for imbalance correction  
- Model evaluation using `classification_report` and `confusion_matrix`  
- Ensemble learning for robust prediction  
- Plotly for interactive data visualization  

## 📌 Dependencies

- `pandas`, `numpy`, `seaborn`, `matplotlib`, `plotly`  
- `scikit-learn`, `xgboost`, `tensorflow`, `imblearn`

## 🧪 How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run any of the model scripts:
   ```bash
   python churn_predictor_final.py
   ```

## 📜 License

This project is provided under the **MIT License**. You are free to use, modify, and distribute the code for academic or commercial purposes.

## 🙏 Acknowledgments

- Dataset from IBM's [Telco Customer Churn](https://www.ibm.com/communities/analytics/watson-analytics-blog/guide-to-sample-datasets/)
- Inspired by open-source churn modeling frameworks and best practices
- Developed using modern Python libraries and open research on model optimization
