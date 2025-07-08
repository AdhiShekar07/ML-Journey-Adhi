import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Generate sample data (noisy sine wave)
np.random.seed(42)
X = np.sort(np.random.rand(50, 1) * 4 * np.pi)
y = np.sin(X).ravel() + np.random.normal(0, 0.3, X.shape[0])

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define different polynomial degrees
degrees = [1, 4, 15]

plt.figure(figsize=(15, 5))
for i, degree in enumerate(degrees):
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Plot
    plt.subplot(1, 3, i + 1)
    plt.scatter(X_train, y_train, color='blue', label='Train')
    plt.scatter(X_test, y_test, color='green', label='Test')
    
    X_plot = np.linspace(0, 4*np.pi, 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)
    plt.plot(X_plot, y_plot, color='red', label=f'Degree {degree}')
    
    plt.title(f'Degree {degree}')
    plt.legend()

plt.suptitle("Bias-Variance Tradeoff: Underfitting vs Overfitting")
plt.tight_layout()
plt.show()
