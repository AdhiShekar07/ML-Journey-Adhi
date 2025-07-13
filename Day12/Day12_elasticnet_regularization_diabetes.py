# elasticnet_regularization_diabetes.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load dataset
data = load_diabetes()
X = data.data
y = data.target
feature_names = data.feature_names

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ElasticNetCV (auto selects best alpha and l1_ratio using cross-validation)
model = ElasticNetCV(cv=5, l1_ratio=[0.1, 0.5, 0.9], random_state=42)
model.fit(X_scaled, y)

# Predictions
y_pred = model.predict(X_scaled)
mse = mean_squared_error(y, y_pred)

# Print results
print("Best Alpha:", model.alpha_)
print("Best L1 Ratio:", model.l1_ratio_)
print("Mean Squared Error:", mse)
print("Coefficients:")
for f, coef in zip(feature_names, model.coef_):
    print(f"{f}: {coef:.4f}")

# Plotting feature importance
plt.figure(figsize=(10, 6))
plt.bar(feature_names, model.coef_)
plt.title("ElasticNet Coefficients (Feature Importance)")
plt.xlabel("Feature")
plt.ylabel("Coefficient Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()
