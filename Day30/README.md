# ⚡ Electricity Consumption Forecasting using LSTM

This project uses a Long Short-Term Memory (LSTM) neural network to forecast hourly electricity demand for the next 7 days based on past consumption data. The model is trained on 3 years of real-world energy usage data to predict future loads for efficient grid management.

---

## 📊 Problem Statement

Electricity providers need to forecast demand accurately to:
- Ensure uninterrupted supply
- Prevent overloads and blackouts
- Optimize energy purchasing and distribution

Traditional models like ARIMA struggle with complex, non-linear, and seasonal patterns in electricity usage. LSTM, a type of Recurrent Neural Network (RNN), is better suited for capturing these time-based dependencies.

---

## 🧠 Model Architecture

- **Input**: 24-hour historical electricity data
- **Layers**:
  - LSTM (50 units)
  - Dropout (0.2)
  - Dense layer (1 unit for output)
- **Output**: Predicted electricity consumption for the next hour
- **Rolling forecast**: Predicts up to 7 days (168 hours) ahead

---

## 📁 Dataset

- Source: Public utility datasets or smart meter readings
- Features:
  - `Datetime`: Timestamp
  - `Consumption (kWh)`: Energy usage at each hour
- Preprocessing:
  - Filled missing values
  - Normalized data
  - Created sliding time windows

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/electricity-lstm-forecast.git
   cd electricity-lstm-forecast
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the training script:
   ```bash
   python lstm_forecasting.py
   ```

4. Visualize the results:
   ```bash
   python visualize_predictions.py
   ```

---

## 📈 Evaluation

- **Loss Function**: Mean Squared Error (MSE)
- **Metric**: Mean Absolute Error (MAE)
- **Rescaled MAE**: ~30 kWh
- Performs well on:
  - Daily and weekly cycles
  - Peak and off-peak variations

---

## 🛠️ Tech Stack

- Python 3.x
- TensorFlow / Keras
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn

---

## 🌟 Applications

- Smart grid load forecasting
- Renewable energy planning
- Peak pricing strategy
- Battery storage optimization

---

## 📌 Extended Description

> A robust LSTM-based forecasting solution to predict hourly electricity demand with high accuracy, enabling power boards to better manage grids and reduce energy costs.

---

## 🧑‍💻 Author

Aditya S | [LinkedIn](https://www.linkedin.com/in/yourprofile)
