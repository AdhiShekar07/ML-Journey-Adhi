import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# 🚫 Model without Feature Scaling
# -------------------------------
model_no_scaling = LogisticRegression(max_iter=10000)
model_no_scaling.fit(X_train, y_train)
preds_no_scaling = model_no_scaling.predict(X_test)
acc_no_scaling = accuracy_score(y_test, preds_no_scaling)
print("🚫 Accuracy without Scaling:", acc_no_scaling)

# -------------------------------------
# ✅ Model with Standardization Scaling
# -------------------------------------
pipeline_std = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=10000))
])
pipeline_std.fit(X_train, y_train)
preds_std = pipeline_std.predict(X_test)
acc_std = accuracy_score(y_test, preds_std)
print("✅ Accuracy with Standardization:", acc_std)

# -------------------------------------
# ✅ Model with Normalization Scaling
# -------------------------------------
pipeline_norm = Pipeline([
    ('scaler', MinMaxScaler()),
    ('classifier', LogisticRegression(max_iter=10000))
])
pipeline_norm.fit(X_train, y_train)
preds_norm = pipeline_norm.predict(X_test)
acc_norm = accuracy_score(y_test, preds_norm)
print("✅ Accuracy with Normalization:", acc_norm)
