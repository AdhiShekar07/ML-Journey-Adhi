# ML Daily Byte – Day 20: Hyperparameter Tuning (Grid vs Random Search)

## 🚀 Concept Overview

Hyperparameter tuning helps us find the **best model configuration** by tweaking parameters like tree depth, learning rate, etc., before training.

Two major techniques:
- **Grid Search**: Exhaustively tries all combinations of parameters.
- **Random Search**: Randomly samples combinations from the parameter space.

---

## 🪐 Real-World Analogy – Cooking Pasta 🍝

- Hyperparameters = flame level, boiling time, salt amount
- Grid Search = trying every combo until it’s perfect
- Random Search = trying smart, random combos to save time

---

## 🛰️ Use Case – Mars Rover Rock Classification

NASA’s rovers classify rock types on Mars using image data.  
Features:
- Color intensity
- Texture variation
- Shape index
- Reflectivity

🔍 Goal: Accurately predict **sedimentary**, **volcanic**, or **mineral-rich** rocks.

🧪 Importance: Helps prioritize **sample collection**, **bandwidth usage**, and potential **biosignature candidates**.

---

## 💻 Code Included

- File: `mars_rover_rock_classification_tuning.py`
- Techniques: `GridSearchCV` and `RandomizedSearchCV` on a simulated dataset
- Model: `RandomForestClassifier`

---

## 📊 Output

- Best hyperparameters found using both techniques
- Classification report comparing model performance

---

## 📎 Hashtags

#MLDailyByte #HyperparameterTuning #GridSearch #RandomSearch  
#MarsRover #SpaceML #ExplainableAI #MachineLearning #AI #Python  
#AdhiExplains #MLForGood #AIForSpace #SciTech
