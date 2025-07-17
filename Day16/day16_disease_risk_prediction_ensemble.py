# 🧪 Disease Risk Prediction using Bagging & Boosting

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1️⃣ Simulated Health Dataset
np.random.seed(42)
n = 500

data = {
    'Age': np.random.randint(25, 75, n),
    'Blood_Pressure': np.random.randint(80, 180, n),
    'Sugar_Level': np.random.randint(70, 200, n),
    'Activity_Level': np.random.randint(1, 10, n),
    'Smoking': np.random.randint(0, 2, n),
    'Clean_Water_Access': np.random.randint(0, 2, n),
    'Family_History': np.random.randint(0, 2, n),
    'Disease_Risk': np.random.choice([0, 1], size=n, p=[0.7, 0.3])  # 0 = No Risk, 1 = Risk
}

df = pd.DataFrame(data)

# 2️⃣ Splitting Data
X = df.drop('Disease_Risk', axis=1)
y = df['Disease_Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# 3️⃣ Bagging Model: Random Forest
bag_model = RandomForestClassifier(n_estimators=100, random_state=42)
bag_model.fit(X_train, y_train)
bag_pred = bag_model.predict(X_test)

print("🎯 Bagging (Random Forest) Results:")
print(classification_report(y_test, bag_pred))

# 4️⃣ Boosting Model: XGBoost
boost_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
boost_model.fit(X_train, y_train)
boost_pred = boost_model.predict(X_test)

print("\n🚀 Boosting (XGBoost) Results:")
print(classification_report(y_test, boost_pred))
