import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy.stats import randint

# Simulated Mars rover image feature dataset
np.random.seed(42)
n = 500
df = pd.DataFrame({
    'color_intensity': np.random.normal(0.5, 0.1, n),
    'texture_variance': np.random.normal(1.2, 0.3, n),
    'shape_index': np.random.normal(0.8, 0.2, n),
    'reflectivity': np.random.normal(0.6, 0.15, n),
    'rock_type': np.random.choice(['sedimentary', 'volcanic', 'mineral-rich'], n)
})

# Encode target
df['rock_type'] = df['rock_type'].astype('category').cat.codes

# Split data
X = df.drop('rock_type', axis=1)
y = df['rock_type']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Random Forest model
model = RandomForestClassifier(random_state=42)

# Grid Search
grid_params = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, None]
}
grid = GridSearchCV(model, grid_params, cv=3)
grid.fit(X_train, y_train)

print("🔍 Grid Search Best Params:", grid.best_params_)
print("📊 Grid Search Report:")
print(classification_report(y_test, grid.predict(X_test)))

# Random Search
random_params = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(2, 10)
}
random = RandomizedSearchCV(model, random_params, n_iter=10, cv=3, random_state=42)
random.fit(X_train, y_train)

print("\n🎯 Random Search Best Params:", random.best_params_)
print("📊 Random Search Report:")
print(classification_report(y_test, random.predict(X_test)))
