import shap
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Seed for reproducibility
np.random.seed(42)

# Simulated dataset
n = 300
df = pd.DataFrame({
    'Payload_Weight_kg': np.random.normal(1200, 200, n),
    'Launch_Window_Hours': np.random.uniform(1, 6, n),
    'Wind_Speed_kmph': np.random.normal(25, 10, n),
    'Fuel_Tank_Temp_C': np.random.normal(30, 5, n),
    'Signal_Calibration_Error': np.random.normal(0.5, 0.2, n),
    'Time_Since_Last_Launch_days': np.random.randint(10, 120, n),
    'Launch_Success': np.random.choice([0, 1], size=n, p=[0.3, 0.7])
})

# Split data
X = df.drop('Launch_Success', axis=1)
y = df['Launch_Success']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Explain with SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Plot summary
shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=True)
