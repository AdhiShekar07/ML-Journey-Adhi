# 📘 Day 14 – Activation Functions (ReLU, Sigmoid, Tanh, GELU)

---

## 🧠 Concept Overview

Activation functions are the **decision-makers in a neural network**.  
They decide whether a neuron should activate (pass signal) or not.

Without them, your model would just be a linear calculator.  
With them, your model can learn **complex patterns**, just like a human brain learns from different situations.

---

## 🍩 Real-Life Analogy – Smart Bakery Decision System

Imagine a smart bakery 🍩 that only bakes when there’s an order.

- No order? Do nothing.  
- Small order? Bake slowly.  
- Big order? Bake full speed!

Each **activation function** is like a rule the bakery follows for deciding **how much to bake**:

| Activation | Behavior | Bakery Logic |
|------------|----------|--------------|
| **ReLU**   | Only positive values pass | “If order > 0, start baking!” |
| **Sigmoid**| Smooth yes/no decision between 0 and 1 | “Maybe bake a little...” |
| **Tanh**   | Balanced output from -1 to +1 | “Bake less or more based on need” |
| **GELU**   | Smart + slightly random activation | “Creative baking decisions” |

---

## 🏥 Real-World Use Case: Medical Diagnosis AI

You’re building a neural network that predicts if a person has a disease based on health data.

- **Sigmoid** in the output layer tells:  
  ➤ “There’s an 82% chance the person has the disease.”

- **ReLU** in hidden layers passes only meaningful symptoms.

- **GELU** is used in models like ChatGPT & BERT for stable learning.

---

## 💻 Code Example: Visualizing Activation Functions

This code shows how ReLU, Sigmoid, and Tanh behave on input data using graphs.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Sample input values
X = np.linspace(-10, 10, 100)

# Apply activation functions
relu = tf.nn.relu(X)
sigmoid = tf.nn.sigmoid(X)
tanh = tf.nn.tanh(X)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(X, relu, label='ReLU', color='red')
plt.plot(X, sigmoid, label='Sigmoid', color='green')
plt.plot(X, tanh, label='Tanh', color='blue')

plt.title("Activation Functions: ReLU vs Sigmoid vs Tanh")
plt.xlabel("Input Value")
plt.ylabel("Activated Output")
plt.grid(True)
plt.legend()
plt.show()
