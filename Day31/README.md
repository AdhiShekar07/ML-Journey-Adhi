# 🎯 Accent Classification Using SMOTE for Imbalanced Data

This project demonstrates how to handle imbalanced datasets using **SMOTE (Synthetic Minority Over-sampling Technique)** while building an **Accent Classification** model using a basic MLP (Multi-layer Perceptron). The goal is to classify speech accents by balancing classes to prevent bias toward dominant ones.

## 📌 Problem Statement

In real-world audio datasets, some accents (e.g., American) may have significantly more samples than others (e.g., Australian or Indian). This class imbalance can mislead ML models into favoring the majority class. This project uses **SMOTE** to synthetically generate samples for minority classes and improve overall model fairness.

## ⚙️ Technologies Used

- Python
- NumPy & Pandas
- Scikit-learn
- imbalanced-learn (for SMOTE)
- Matplotlib / Seaborn (optional for visualization)

## 📂 Dataset Description

Assume the dataset contains:
- **features** extracted from audio (e.g., MFCCs)
- **target** column: different **accent categories**

Sample structure:
| mfcc1 | mfcc2 | ... | accent   |
|-------|-------|-----|----------|
| 0.23  | 1.42  | ... | American |
| 0.12  | 0.97  | ... | Indian   |
| ...   | ...   | ... | ...      |

## 📈 Pipeline Overview

1. Load and preprocess audio features
2. Apply **SMOTE** to balance accent classes
3. Train a **simple MLP classifier**
4. Evaluate using accuracy, precision, recall, and confusion matrix

## ✅ Results

Using SMOTE improved minority class recall and made the model perform better across all classes instead of favoring only majority accents.

## 🚀 Future Improvements

- Use real MFCC audio feature extraction
- Replace MLP with CNN/LSTM for better audio modeling
- Try other imbalance-handling methods like **ADASYN** or **class weights**

## 📁 File Structure

- `accent_classification_with_smote.py` – Full code
- `README.md` – Documentation and instructions

---

#machinelearning #imbalanceddata #smote #audioclassification #accentdetection #deeplearning
