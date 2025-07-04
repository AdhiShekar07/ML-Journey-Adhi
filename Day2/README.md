# 📘 ML Daily Byte – Day 2: Data Splitting & Spam Detection

Welcome to **Day 2** of my Machine Learning learning series.  
Today, I explored one of the most important concepts in ML development: **splitting data** correctly — and applied that knowledge to a practical **Spam Email Classifier** project using the Naive Bayes algorithm.

---

## 🔍 Why Split Data?

Before training any ML model, we need to divide our dataset so that we can evaluate how well our model performs on **unseen data**. This helps prevent **overfitting** and improves model **generalization**.

### ✨ Real-Life Analogy:
Think of preparing for an exam:
- 📚 **Training Set** – notes and materials you study
- 📝 **Validation Set** – practice tests to adjust your learning
- 🎓 **Test Set** – your final exam that tests your understanding

---

## 🔄 Common Data Split Ratios

| Set          | Purpose                      | Typical Split |
|--------------|------------------------------|----------------|
| Training     | Learn patterns from data      | 60–80%         |
| Validation   | Tune model performance        | 10–20%         |
| Testing      | Final unbiased evaluation     | 10–20%         |

---

## 🌼 Example 1: Iris Dataset Split

A classic dataset for beginners in ML. Here’s a basic example of how to split it:

```python
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training data size:", len(X_train))
print("Testing data size:", len(X_test))
