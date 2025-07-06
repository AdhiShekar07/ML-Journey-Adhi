# 💳 ML Daily Byte – Day 4: ROC vs PR Curves in Credit Card Fraud Detection

Today I tackled a real-world, high-impact machine learning task — **detecting fraudulent credit card transactions** — and evaluated the model using two powerful tools: **ROC Curve** and **Precision-Recall Curve**.

## 🎯 Problem Overview

- Goal: Predict whether a transaction is **fraudulent or not**
- Challenge: Fraud is **very rare** (imbalanced dataset)
- Solution: Evaluate model using:
  - **ROC Curve** — how well can it distinguish?
  - **PR Curve** — how confident are its fraud predictions?

---

## 🔎 Key Metrics

| Metric       | What It Tells You                  | Use When...               |
|--------------|------------------------------------|---------------------------|
| ROC Curve    | TPR vs FPR                         | Classes are balanced      |
| PR Curve     | Precision vs Recall                | Rare class is important   |
| AUC          | Area under ROC (higher = better)   | Overall model score       |

---

## 🧠 Simple Example Code

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

# Simulate 1000 transactions (95% genuine, 5% fraud)
X, y = make_classification(n_samples=1000, n_features=10, weights=[0.95, 0.05], random_state=42)

# Train/Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)
y_scores = model.predict_proba(X_test)[:, 1]

# Plot ROC & PR Curves
fpr, tpr, _ = roc_curve(y_test, y_scores)
precision, recall, _ = precision_recall_curve(y_test, y_scores)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(fpr, tpr, label='ROC AUC')
plt.xlabel('FPR')
plt.ylabel('TPR (Recall)')
plt.title('ROC Curve')
plt.legend()

plt.subplot(1,2,2)
plt.plot(recall, precision, color='green')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.tight_layout()
plt.show()

📊 Observations
ROC Curve: Visualizes how well the model separates fraud vs normal

PR Curve: Focuses on how reliable the fraud predictions are

AUC: Shows overall model power

📂 File Reference
Main Code: Day4_ROC_PR_CreditCardFraud.py

Tip: Also try creating a DebitCard_Fraud_Detection.py with slightly different imbalance and more features!

🧠 Key Takeaways
Always use PR Curve for rare events like fraud/disease detection

ROC can mislead you when the dataset is highly imbalanced

Visuals help interpret model behavior beyond accuracy

🔗 Resources
ROC vs PR Curves (sklearn docs)

Precision vs Recall (Article)

#MachineLearning #ROC #PrecisionRecall #CreditCardFraud #ImbalancedData #AUC #ScikitLearn #Python #LogisticRegression #MLDailyByte

