# 📘 ML Daily Byte – Day 3: Confusion Matrix & Evaluation Metrics

Today I learned how to evaluate ML classification models using the **Confusion Matrix** and key metrics like **Accuracy**, **Precision**, **Recall**, and **F1-Score**.

---

## 🔹 What is a Confusion Matrix?

A 2x2 table that compares the predicted results with actual values in binary classification.

|                | Predicted: Positive | Predicted: Negative |
|----------------|---------------------|---------------------|
| Actual: Positive | True Positive (TP)  | False Negative (FN) |
| Actual: Negative | False Positive (FP) | True Negative (TN)  |

---

## 🌍 Real-Life Analogy: COVID-19 Test

- TP: Sick person correctly detected as sick ✅  
- TN: Healthy person correctly detected as healthy ✅  
- FP: Healthy person wrongly detected as sick ❌  
- FN: Sick person wrongly detected as healthy ❌  

---

## 📏 Key Metrics

| Metric     | Formula                          | Use When                                 |
|------------|----------------------------------|------------------------------------------|
| Accuracy   | (TP + TN) / Total                | Classes are balanced                     |
| Precision  | TP / (TP + FP)                   | False positives are dangerous            |
| Recall     | TP / (TP + FN)                   | Missing real positives is dangerous      |
| F1-Score   | 2 * (P * R) / (P + R)             | Balance between Precision & Recall       |

---

## 🧑‍💻 Code Example: Breast Cancer Prediction

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\\n", cm)

print("\\nClassification Report:\\n")
print(classification_report(y_test, y_pred, target_names=data.target_names))
