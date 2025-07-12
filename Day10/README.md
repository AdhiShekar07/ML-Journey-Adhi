# 🧠 ML Daily Byte – Day 10: Feature Scaling with StandardScaler

## 📌 Concept:
Feature Scaling is a technique to normalize the range of independent variables. It helps prevent certain features from dominating others due to scale differences.

In real-world datasets, some features like "Salary" can be in thousands, while others like "Age" are in tens. Without scaling, ML models may give more importance to larger numerical ranges.

---

## 📚 Why It Matters:
- Distance-based models like **KNN**, **K-Means**, and **SVM** are sensitive to feature magnitudes.
- Gradient-based models like **Logistic Regression** and **Neural Networks** converge faster with scaled features.
- Feature scaling improves **model accuracy** and **training stability**.

---

## 🛠️ What I Did:
- Created a NumPy array with features: Age and Salary.
- Used `StandardScaler` from `sklearn.preprocessing`.
- Transformed the data so that it has **zero mean** and **unit variance**.

---

## 🧪 Sample Code:
```python
from sklearn.preprocessing import StandardScaler
import numpy as np

X = np.array([[25, 50000], [32, 60000], [47, 80000], [51, 90000], [62, 120000]])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
