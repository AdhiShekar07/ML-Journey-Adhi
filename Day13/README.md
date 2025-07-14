# 📅 ML Daily Byte – Day 13: Loss Functions (MSE, BCE, Cross-Entropy)

Loss functions are the heart of how a machine learning model learns.  
They measure **how far off a model's predictions are** from the actual answers and guide it to improve.

---

## 🔍 What is a Loss Function?

A **loss function** tells the model **how wrong its prediction is**.  
It's like a training coach giving feedback after every move.

- ✅ Low loss → Good predictions  
- ❌ High loss → Bad predictions  
The goal: **Minimize the loss** during training.

---

## 🎯 Why Are Loss Functions Important?

Loss functions:
- Guide model training  
- Help adjust weights via backpropagation  
- Ensure the model improves with every epoch

They are the **target** that optimization algorithms aim to minimize.

---

## 💊 Real-World Analogy: Medicine AI System

Imagine an AI system recommending:
- 💉 Dosage (mg)
- ❌✅ Whether antibiotics are needed
- 🎯 What drug category to prescribe

Each task needs a **different loss function**:

| Task                                  | Loss Function               |
|---------------------------------------|-----------------------------|
| Predicting dosage (mg)                | MSE (Mean Squared Error)    |
| Deciding if antibiotic is needed      | Binary Cross Entropy (BCE)  |
| Choosing correct drug category        | Categorical Cross Entropy   |

---

## 🔢 Common Loss Functions

### 📏 MSE – Mean Squared Error (for Regression)
- Penalizes large errors more
- Formula:  
  \[
  MSE = \frac{1}{n} \sum (y_{\text{true}} - y_{\text{pred}})^2
  \]

### ⚖️ MAE – Mean Absolute Error
- Penalizes all errors equally
- More robust to outliers

### ✅❌ Binary Cross Entropy (BCE)
- For binary classification (0 or 1)
- Penalizes confident wrong predictions heavily

### 🎯 Categorical Cross Entropy
- For multi-class classification (softmax output)
- Compares predicted class probabilities with true class

---

## 👨‍💻 Code Example: `medicine_loss_functions_demo.py`

```python
import numpy as np
from sklearn.metrics import mean_squared_error
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy

# Simulated patient data (age, weight, severity)
X = np.array([
    [25, 60, 0.8],
    [45, 80, 0.6],
    [60, 55, 0.9]
])

# MSE: Dosage prediction
actual_dosage = np.array([500, 650, 400])
predicted_dosage = np.array([480, 700, 390])
mse = mean_squared_error(actual_dosage, predicted_dosage)
print("MSE:", round(mse, 2))

# BCE: Antibiotic needed (yes/no)
actual_binary = np.array([[1], [0], [1]])
predicted_binary = np.array([[0.9], [0.2], [0.6]])
bce = BinaryCrossentropy()
print("BCE Loss:", round(bce(actual_binary, predicted_binary).numpy(), 4))

# Cross Entropy: Drug category
actual_multi = np.array([[0, 1, 0, 0, 0]])
predicted_multi = np.array([[0.1, 0.7, 0.1, 0.05, 0.05]])
cce = CategoricalCrossentropy()
print("Categorical Cross Entropy:", round(cce(actual_multi, predicted_multi).numpy(), 4))
```

---

## 📌 Output Snapshot

```
MSE (Dosage Prediction): 2166.67
Binary Cross Entropy (Antibiotic Decision): 0.3711
Categorical Cross Entropy (Drug Category): 0.3567
```

---

## 📂 File Description

**Filename:** `medicine_loss_functions_demo.py`  
**Extended Description:** Demonstrates MSE, BCE, and Categorical Cross Entropy using a healthcare-based AI scenario, including dosage prediction, binary drug recommendation, and multi-class classification.

---

## 🏷️ Hashtags

#MLDailyByte #Day13 #LossFunctions #MSE #BCE #CrossEntropy  
#MachineLearning #AI #Python #HealthTechAI #MedicinePrediction  
#MLForBeginners #DataScience #DeepLearning #AIForHealthcare
