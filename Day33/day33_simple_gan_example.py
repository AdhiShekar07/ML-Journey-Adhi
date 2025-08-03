import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Generator model
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(16, activation="relu", input_shape=(10,)),
        layers.Dense(1, activation="linear")  # Output is a single float (1D real-like value)
    ])
    return model

# Discriminator model
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Dense(16, activation="relu", input_shape=(1,)),
        layers.Dense(1, activation="sigmoid")  # Output: probability of being real
    ])
    return model

# Loss and optimizers
loss_fn = tf.keras.losses.BinaryCrossentropy()
gen_optimizer = tf.keras.optimizers.Adam(0.001)
disc_optimizer = tf.keras.optimizers.Adam(0.001)

# Build models
generator = build_generator()
discriminator = build_discriminator()

# Training loop
def train_gan(epochs=1000):
    for epoch in range(epochs):
        # === Train Discriminator ===
        real_data = np.random.normal(4, 1, size=(16, 1))  # Real data ~ N(4,1)
        fake_input = np.random.normal(0, 1, size=(16, 10))  # Random noise
        fake_data = generator.predict(fake_input)

        # Labels
        real_labels = np.ones((16, 1))
        fake_labels = np.zeros((16, 1))

        # Train on real and fake
        with tf.GradientTape() as tape:
            pred_real = discriminator(real_data)
            pred_fake = discriminator(fake_data)
            d_loss = loss_fn(real_labels, pred_real) + loss_fn(fake_labels, pred_fake)

        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        disc_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        # === Train Generator ===
        noise = np.random.normal(0, 1, size=(16, 10))
        misleading_labels = np.ones((16, 1))  # Try to fool the discriminator

        with tf.GradientTape() as tape:
            generated = generator(noise)
            pred = discriminator(generated)
            g_loss = loss_fn(misleading_labels, pred)

        grads = tape.gradient(g_loss, generator.trainable_variables)
        gen_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: D_loss={d_loss.numpy():.4f}, G_loss={g_loss.numpy():.4f}")

# Run training
train_gan(epochs=1000)

# Generate and plot samples
test_noise = np.random.normal(0, 1, size=(100, 10))
generated_samples = generator.predict(test_noise)

plt.hist(generated_samples, bins=20, label="Generated", alpha=0.6)
plt.hist(np.random.normal(4, 1, 100), bins=20, label="Real", alpha=0.6)
plt.legend()
plt.title("Real vs Generated Data")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
