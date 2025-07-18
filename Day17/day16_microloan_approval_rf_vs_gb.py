import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# 1️⃣ Simulate dataset for rural women entrepreneurs
np.random.seed(10)
n = 500
df = pd.DataFrame({
    'Credit_History': np.random.choice(['good', 'bad'], size=n, p=[0.7, 0.3]),
    'Married': np.random.choice(['yes', 'no'], size=n, p=[0.8, 0.2]),
    'Business_Type': np.random.choice(['handloom', 'farming', 'food_stall'], size=n),
    'Owns_Home': np.random.choice(['yes', 'no'], size=n, p=[0.6, 0.4]),
    'Mobile_Type': np.random.choice(['smartphone', 'basic', 'none'], size=n),
    'Monthly_Savings': np.random.randint(500, 8000, size=n),
    'Dependents': np.random.randint(0, 5, size=n),
    'Loan_Approved': np.random.choice([0, 1], size=n, p=[0.4, 0.6])
})

# 2️⃣ Encode categorical columns
label_cols = ['Credit_History', 'Married', 'Business_Type', 'Owns_Home', 'Mobile_Type']
df_encoded = df.copy()
le = LabelEncoder()
for col in label_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])

# 3️⃣ Train-test split
X = df_encoded.drop('Loan_Approved', axis=1)
y = df_encoded['Loan_Approved']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# 4️⃣ Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

print("🌲 Random Forest Results:")
print(classification_report(y_test, rf_preds))

# 5️⃣ Gradient Boosting Model
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb.fit(X_train, y_train)
gb_preds = gb.predict(X_test)

print("\n🚀 Gradient Boosting Results:")
print(classification_report(y_test, gb_preds))
