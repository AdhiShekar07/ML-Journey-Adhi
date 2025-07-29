# Student Performance Predictor using LSTM

🎯 This project demonstrates how an LSTM (Long Short-Term Memory) network can be used to predict a student's future performance based on historical test scores and attendance data.

## 📌 Project Highlights
- Implements an LSTM model using PyTorch.
- Input: Sequence of test scores and attendance over time.
- Output: Predicted next test score.
- Demonstrates temporal sequence learning in the education domain.

## 🚀 How It Works
1. Student data is normalized and reshaped for LSTM input.
2. The LSTM learns temporal dependencies in the data.
3. A fully connected layer outputs the predicted next score.

## 🧠 Concept Used
- LSTM (Long Short-Term Memory)
- Sequence modeling
- Time-series prediction
- PyTorch framework

## 🔧 Libraries Used
- torch
- numpy

## 📁 File Structure
- `StudentPerformancePredictorLSTM.py`: Contains the PyTorch LSTM implementation.

## ✅ Usage
```bash
python StudentPerformancePredictorLSTM.py
