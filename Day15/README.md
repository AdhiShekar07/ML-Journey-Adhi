# 📘 Day 15 – Decision Trees & Gini/Entropy

---

## 🧠 Concept Overview

A **Decision Tree** is a flowchart-like model that splits data based on features to make decisions.  
It works by asking a series of questions and branching based on the answers until a final decision (label) is made.

Two popular methods to measure the **quality of a split**:
- **Gini Impurity**: Measures how mixed the data is (0 = pure)
- **Entropy**: Measures randomness or uncertainty in the data

The goal is to split the data so that each branch becomes as **pure** as possible.

---

## 🍩 Simple Analogy – Ice Cream Decision Tree

Imagine you're helping customers choose ice cream:

1. Is it hot outside?  
2. Do you like chocolate?  
3. Cone or cup?

Each question filters choices until the perfect ice cream is chosen.  
That’s exactly what a **decision tree** does – narrowing down to the right answer step by step.

---

## 🚜 Real-World Example – Crop Disease Diagnosis for Farmers

You’re building an AI tool to detect **crop diseases** early.

### ✅ Features:
- Leaf color (yellow/brown or normal)
- Presence of black spots
- Recent rainfall
- Soil moisture level

### 🧪 Target:
- Fungal
- Healthy

The model learns patterns from these features and uses a **decision tree** to predict the disease class.

---

## 💻 Code Example

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Sample dataset
data = {
    'Leaf_Color': [1, 0, 1, 1, 0, 0, 1],
    'Spots': [1, 0, 1, 0, 1, 0, 1],
    'Rainfall_mm': [60, 20, 55, 10, 80, 5, 70],
    'Soil_Moisture': [30, 70, 40, 90, 20, 85, 35],
    'Disease': ['Fungal', 'Healthy', 'Fungal', 'Healthy', 'Fungal', 'Healthy', 'Fungal']
}

df = pd.DataFrame(data)
X = df[['Leaf_Color', 'Spots', 'Rainfall_mm', 'Soil_Moisture']]
y = df['Disease']

# Train Decision Tree using Gini
model = DecisionTreeClassifier(criterion='gini')
model.fit(X, y)

# Visualize the tree
plt.figure(figsize=(12, 6))
tree.plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True)
plt.title("Crop Disease Diagnosis - Decision Tree")
plt.show()
