import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Simulated flight data
np.random.seed(42)
n = 500

df = pd.DataFrame({
    'Engine_Temp': np.random.normal(90, 10, n),
    'Sensor_Errors': np.random.poisson(2, n),
    'Altitude_Change': np.random.normal(0.5, 0.2, n),
    'Wind_Turbulence': np.random.uniform(0, 1, n),
    'Fuel_Imbalance': np.random.normal(5, 2, n),
    'Pilot_Hours': np.random.randint(1, 15, n),
    'Weather_Severity': np.random.randint(0, 3, n),
    'Time_Since_Maintenance': np.random.randint(1, 100, n),
    'Crash_Risk': np.random.choice([0, 1], size=n, p=[0.85, 0.15])
})

# Split
X = df.drop('Crash_Risk', axis=1)
y = df['Crash_Risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# Train
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
preds = rf.predict(X_test)
print("🔎 Classification Report:")
print(classification_report(y_test, preds))

# Feature importance
importances = rf.feature_importances_
feature_names = X.columns
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp_df, x='Importance', y='Feature', palette='coolwarm')
plt.title('✈️ Aircraft Crash Risk – Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()
