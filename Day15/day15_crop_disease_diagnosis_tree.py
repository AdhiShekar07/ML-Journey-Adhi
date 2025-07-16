import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Sample dataset for crop disease diagnosis
data = {
    'Leaf_Color': [1, 0, 1, 1, 0, 0, 1],     # 1 = yellow/brown, 0 = normal
    'Spots': [1, 0, 1, 0, 1, 0, 1],          # 1 = black spots present, 0 = no
    'Rainfall_mm': [60, 20, 55, 10, 80, 5, 70],
    'Soil_Moisture': [30, 70, 40, 90, 20, 85, 35],
    'Disease': ['Fungal', 'Healthy', 'Fungal', 'Healthy', 'Fungal', 'Healthy', 'Fungal']
}

df = pd.DataFrame(data)

# Features and labels
X = df[['Leaf_Color', 'Spots', 'Rainfall_mm', 'Soil_Moisture']]
y = df['Disease']

# Build Decision Tree Classifier with Gini
model = DecisionTreeClassifier(criterion='gini')
model.fit(X, y)

# Visualize the decision tree
plt.figure(figsize=(12, 6))
tree.plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True)
plt.title("Crop Disease Diagnosis - Decision Tree")
plt.show()
