# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Step 1: Create a small student dataset manually
# Each row represents a student
data = {
    'attendance': [85, 45, 72, 60, 30, 95, 40, 78, 55, 25],  # in %
    'internals':  [80, 40, 75, 65, 35, 90, 42, 70, 50, 20],  # internal marks
    'backlogs':   [0, 3, 1, 2, 5, 0, 4, 1, 3, 6],            # number of subjects failed
    'dropout':    [0, 1, 0, 0, 1, 0, 1, 0, 1, 1]             # 1 = dropped out, 0 = continued
}

# Step 2: Convert the dictionary into a DataFrame
df = pd.DataFrame(data)

# Step 3: Split data into input features (X) and target (y)
X = df[['attendance', 'internals', 'backlogs']]  # Features we use to predict
y = df['dropout']                                # What we are predicting

# Step 4: Scale the feature values for better model performance
# Scaling makes all features lie on the same range
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Apply standardization to X

# Step 5: Create a Logistic Regression model (good for binary classification)
model = LogisticRegression()

# Step 6: Define Stratified K-Fold (to ensure each fold has the same dropout ratio)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# Step 7: Perform cross validation using the model and stratified folds
scores = cross_val_score(model, X_scaled, y, cv=skf)

# Step 8: Display each fold's accuracy
print("Stratified K-Fold Scores:", scores)

# Step 9: Display average accuracy across all folds
print("Average Accuracy:", round(np.mean(scores), 2))
