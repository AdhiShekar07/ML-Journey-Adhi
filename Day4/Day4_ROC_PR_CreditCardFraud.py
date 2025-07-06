# 💳 Credit Card Fraud Detection – ROC vs PR Curve

# Goal: Classify whether a transaction is fraudulent or not using logistic regression
# Evaluate the classifier using ROC and Precision-Recall Curves

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

# 1. Create a simulated imbalanced dataset (like fraud detection)
X, y = make_classification(n_samples=1000,        # total records
                           n_features=10,         # 10 input features
                           n_classes=2,           # binary output (fraud / not)
                           weights=[0.95, 0.05],  # 5% fraud cases
                           random_state=42)       # same result every run

# 2. Split into train and test sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)

# 3. Train a logistic regression classifier
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Get predicted probabilities for the positive class (fraud = 1)
y_scores = model.predict_proba(X_test)[:, 1]

# 5. Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# 6. Compute Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_scores)

# 7. Plot both ROC and PR curves
plt.figure(figsize=(12, 5))

# ROC Curve Plot
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color='blue')
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend()

# PR Curve Plot
plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='green')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")

plt.tight_layout()
plt.show()
