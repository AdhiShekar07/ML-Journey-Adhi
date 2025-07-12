# 📘 Day 11 – Gradient Descent Variants (SGD, RMSProp, Adam)

---

## 🧠 Concept Explanation

**Gradient Descent** is a method used to minimize error in a machine learning model by adjusting weights step-by-step, moving towards the best possible model (lowest error).

When the model guesses wrong, it calculates how wrong it was (called loss), then takes a small step to improve.

### 🔁 Why Variants?

Gradient Descent has multiple **variants** for better and faster learning:

| Variant | Description |
|--------|-------------|
| **SGD (Stochastic Gradient Descent)** | Updates weights using one data point at a time – fast but noisy. |
| **RMSProp** | Adjusts the learning rate based on past gradients – smoother. |
| **Adam** | Combines memory and momentum – smart, fast, and widely used. |

---

## 🍔 Real-World Analogy

### **The Delivery Robot Story**

Imagine you're training a robot to deliver a parcel across a town.

- First try: It takes a random route (slow and wrong).
- Then it learns from the mistake and adjusts.
- It keeps adjusting (like gradient steps) until it finds the shortest route.

This is just like **Gradient Descent**: small adjustments toward a better answer.

**Adam Optimizer** = Smart robot that remembers past routes and speeds up intelligently 🚀

---

## 🏡 Practical Use Case: Rural Job Matching Bot

Build an AI bot that recommends jobs to rural students based on:
- Skills
- Education level
- Available resources

Using **Gradient Descent (Adam)**, the bot learns from past matches and improves accuracy, bringing **real-world value** to students.

---

## 💻 Sample Code: Predict Student Scores

```python
import tensorflow as tf
import numpy as np

# Input: Hours studied
X = np.array([1, 2, 3, 4, 5], dtype=float)
# Output: Exam scores
y = np.array([45, 50, 65, 70, 75], dtype=float)

# Model with 1 neuron
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compile with Adam optimizer
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
              loss='mean_squared_error')

# Train the model
history = model.fit(X, y, epochs=100, verbose=0)

# Predict
print("Predicted Score for 6 hours studied:", model.predict([6.0])[0][0])
