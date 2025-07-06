# 💳 ML Daily Byte – Day 4: ROC vs PR Curves in Credit Card Fraud Detection

Today I explored a real-world machine learning task — **credit card fraud detection** — and evaluated the model using two powerful techniques: **ROC Curve** and **Precision-Recall Curve**. These curves give insights into model performance especially when **fraud cases are rare** (imbalanced data).

![Fraud vs Genuine](images/fraud_distribution.png)
![ROC vs PR](images/roc_pr_curve.png)

---

## 🎯 Problem Overview

- Predict whether a transaction is **fraudulent or not**
- Fraud is **rare** → dataset is **imbalanced**
- Accuracy is misleading → we use:
  - **ROC Curve**: How well the model distinguishes fraud vs non-fraud
  - **PR Curve**: How reliable fraud predictions are

---

## 🔎 Key Metrics

| Metric       | What It Tells You                  | Use When...               |
|--------------|------------------------------------|---------------------------|
| ROC Curve    | TPR vs FPR                         | Balanced classes          |
| PR Curve     | Precision vs Recall                | Positive class is rare    |
| AUC          | Area under ROC (0.5–1.0)           | Model's discrimination    |

---

## 🧠 Code Example

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Simulate 1000 transactions: 95% normal, 5% fraud
X, y = make_classification(n_samples=1000, n_features=10, weights=[0.95, 0.05], random_state=42)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict fraud probabilities
y_scores = model.predict_proba(X_test)[:, 1]

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_scores)
precision, recall, _ = precision_recall_curve(y_test, y_scores)

# Plot
plt.figure(figsize=(12, 5))

# ROC
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label='ROC AUC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend()

# PR
plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='green')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')

plt.tight_layout()
plt.show()
