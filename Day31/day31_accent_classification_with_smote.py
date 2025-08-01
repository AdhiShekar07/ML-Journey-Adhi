import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Step 1: Simulated MFCC feature dataset with imbalance
# Let's assume 13 MFCC features per voice sample
np.random.seed(42)
standard_american = np.random.normal(loc=0.0, scale=1.0, size=(500, 13))
south_indian = np.random.normal(loc=1.0, scale=1.2, size=(40, 13))   # underrepresented
nigerian = np.random.normal(loc=-1.0, scale=1.1, size=(30, 13))       # underrepresented

X = np.vstack((standard_american, south_indian, nigerian))
y = np.array(['american'] * 500 + ['south_indian'] * 40 + ['nigerian'] * 30)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded)

# Step 3: SMOTE to balance minority accent samples
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Step 4: Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train classifier
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train_smote)

# Step 6: Evaluate performance
y_pred = model.predict(X_test_scaled)
print("🔍 Classification Report After SMOTE:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))
