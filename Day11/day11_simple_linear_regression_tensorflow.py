# day11.py
# Concept: Simple Linear Regression using TensorFlow

import tensorflow as tf
import numpy as np

# Sample Data: Hours studied vs Exam scores
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([45, 50, 65, 70, 75], dtype=float)

# Model: 1 neuron
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compile using Adam (a smart Gradient Descent variant)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
              loss='mean_squared_error')

# Train the model
history = model.fit(X, y, epochs=100, verbose=0)

# Predict score for 6 hours studied
print("Predicted Score:", model.predict([6.0])[0][0])
