import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate some sample input data
X = np.linspace(-10, 10, 100)

# Apply different activation functions
relu = tf.nn.relu(X)
sigmoid = tf.nn.sigmoid(X)
tanh = tf.nn.tanh(X)

# Plotting the activation functions
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
