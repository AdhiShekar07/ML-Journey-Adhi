# 🎓 ML Daily Byte – Day 5: Student Dropout Prediction with Stratified K-Fold CV

In today’s project, I tackled a **real-world educational problem** — predicting whether a student might drop out based on early academic indicators.

We used **Stratified K-Fold Cross Validation** to fairly evaluate our model, especially because dropout cases are rare (imbalanced class problem).

---

## 🎯 Problem Statement

Can we predict student dropout risk based on:
- ✅ Attendance %
- ✅ Internal Marks
- ✅ Number of Backlogs

This helps institutions take **early intervention** before it's too late.

---

## 🔄 Why Cross Validation?

A single train-test split may give **misleading accuracy** due to bias or imbalance.

✅ So we use **Stratified K-Fold CV**:
- Keeps the dropout ratio the same in every fold
- Ensures a **fair and balanced evaluation**
- Prevents **overfitting to one random split**

---

## 🧠 Real-World Analogy

> Don’t judge food by just one bite.  
> Taste it from multiple spoons (K-Fold), and make sure each has rice + masala (Stratified)!

---

## 🧪 Model & Code

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Simulated student data
data = {
    'attendance': [85, 45, 72, 60, 30, 95, 40, 78, 55, 25],
    'internals':  [80, 40, 75, 65, 35, 90, 42, 70, 50, 20],
    'backlogs':   [0, 3, 1, 2, 5, 0, 4, 1, 3, 6],
    'dropout':    [0, 1, 0, 0, 1, 0, 1, 0, 1, 1]
}

df = pd.DataFrame(data)
X = df[['attendance', 'internals', 'backlogs']]
y = df['dropout']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

scores = cross_val_score(model, X_scaled, y, cv=skf)

print("Stratified K-Fold Scores:", scores)
print("Average Accuracy:", round(np.mean(scores), 2))
```

---

## 📊 Output (Sample)

```
Stratified K-Fold Scores: [1.  1.  0.5  1.  0.5]
Average Accuracy: 0.8
```

---

## 📌 Observations

- Some folds scored perfectly, others dropped to 50%
- This confirms how **fold variability** affects performance
- On average, model is **80% accurate** with very little data

---

## 📂 File Reference

- 📁 `Day5_StudentDropout_StratifedKFold.py`

---

## 🚀 Key Takeaways

- Use **Stratified K-Fold** for fair evaluation on imbalanced datasets
- Even simple models can predict **important educational outcomes**
- ML can help institutions **reduce dropout rates early**

---

## 🔗 Resources

- [StratifiedKFold – sklearn docs](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)
- [Why Cross Validation Matters – Blog](https://towardsdatascience.com/cross-validation-explained-evaluating-estimator-performance-e51e5430ff1e)

---

#MachineLearning #StudentAnalytics #CrossValidation #StratifiedKFold #LogisticRegression  
#MLDailyByte #DropoutPrediction #EducationAI #DataScience #Python #Sklearn
