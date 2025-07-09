# 🔐 ML Daily Byte – Day 7: Data Leakage in Machine Learning

Data leakage is one of the **most dangerous and sneaky mistakes** in machine learning.  
It leads to **inflated performance metrics** and models that **fail in the real world**.

---

## 🧠 What is Data Leakage?

**Data Leakage** occurs when the training data contains information that **would not be available at prediction time**.  
This causes the model to **cheat unknowingly**, resulting in misleadingly high accuracy.

---

## 🚨 Why It’s a Problem

- Makes the model look "perfect" on test data
- Fails badly in real-world scenarios
- Reinforces bias or past decisions without true learning

---

## 🏥 Real-World Analogy: Health Model

Imagine building a model to **predict heart attacks**, but you include a column `treatment_given` — which is only recorded **after the diagnosis**.  
The model just learns:  
> "If treatment was given → must be a heart attack!"  
Not smart — that’s **leakage**.

---

## 🎓 Social Analogy: Education Loan Model

Suppose you’re building a model to predict **education loan approval** for students.

If you include the feature `loan_approved` (the actual outcome) in your training data:

- Model says “yes” only because the past system did  
- It doesn't learn *why* students get approved  
- May unfairly deny deserving new applicants

📌 **Impact**: Affects low-income students. Ethical ML starts with clean data!

---

## 🔍 Types of Data Leakage

| Type                     | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| **Target Leakage**       | Features contain info that won’t be known at prediction time                |
| **Train-Test Contamination** | Test data leaks into training pipeline during preprocessing or scaling     |
| **Temporal Leakage**     | Model sees future data when predicting earlier events                       |

---

## 👨‍💻 Code Example: Leaky vs Clean Model

```python
# Simulated dataset
data = {
    'age': [50, 60, 45, 35, 70, 30],
    'bp': [120, 140, 130, 110, 150, 115],
    'cholesterol': [200, 240, 180, 170, 250, 190],
    'diagnosis': [1, 1, 0, 0, 1, 0],
    'treatment_given': [1, 1, 0, 0, 1, 0]  # 🚨 Future info!
}
df = pd.DataFrame(data)

# Leaky features
X_leak = df[['age', 'bp', 'cholesterol', 'treatment_given']]
y = df['diagnosis']

# Clean features
X_clean = df[['age', 'bp', 'cholesterol']]

**🧠 Key Takeaways**
Don’t include features unavailable at real-world prediction time

Always split train/test before preprocessing

Watch out for timestamp and target leakage

Ask: “Would I know this at prediction time?”
**🔗 Resources**
Kaggle: Avoiding Data Leakage

Scikit-learn Guide on Data Preprocessing

Blog: Data Leakage in ML

#DataLeakage #EthicalAI #MLDailyByte #EducationAI #ModelValidation
#TrainTestSplit #MachineLearning #Python #DataScience #Generalization
