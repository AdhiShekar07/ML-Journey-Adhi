import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Simulated Education Loan Dataset
data = {
    'student_marks': [85, 70, 60, 90, 40, 75],
    'parent_income': [50000, 30000, 20000, 60000, 15000, 35000],
    'college_rank': [1, 2, 3, 1, 3, 2],
    'loan_amount': [2, 1.5, 1, 2.5, 1, 1.2],
    'loan_approved': [1, 1, 0, 1, 0, 1]  # This is the TARGET
}

df = pd.DataFrame(data)

# -------------------------
# 🚨 Leaky Version (wrong)
# -------------------------
X_leaky = df[['student_marks', 'parent_income', 'college_rank', 'loan_amount', 'loan_approved']]  # <- 🚨 Target included!
y_leaky = df['loan_approved']

X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_leaky, y_leaky, test_size=0.3, random_state=42)

model_leaky = LogisticRegression()
model_leaky.fit(X_train_l, y_train_l)
preds_leaky = model_leaky.predict(X_test_l)

print("🚨 Leaky Model Accuracy:", accuracy_score(y_test_l, preds_leaky))  # Might show perfect or near-perfect

# -------------------------
# ✅ Clean Version (correct)
# -------------------------
X_clean = df[['student_marks', 'parent_income', 'college_rank', 'loan_amount']]  # ✅ No target info in features
y_clean = df['loan_approved']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clean, y_clean, test_size=0.3, random_state=42)

model_clean = LogisticRegression()
model_clean.fit(X_train_c, y_train_c)
preds_clean = model_clean.predict(X_test_c)

print("✅ Clean Model Accuracy:", accuracy_score(y_test_c, preds_clean))  # More realistic
