import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt

# Step 1: Simulate player trait data (1000 players, 3 features)
np.random.seed(42)
speed = np.random.normal(70, 10, 1000)     # average speed
agility = np.random.normal(65, 12, 1000)   # average agility
strength = np.random.normal(75, 8, 1000)   # average strength

player_data = np.stack([speed, agility, strength], axis=1)
player_data = (player_data - player_data.mean(axis=0)) / player_data.std(axis=0)  # normalize

# Step 2: Define VAE architecture
input_dim = 3
latent_dim = 2

# Encoder
inputs = tf.keras.Input(shape=(input_dim,))
x = layers.Dense(16, activation='relu')(inputs)
z_mean = layers.Dense(latent_dim)(x)
z_log_var = layers.Dense(latent_dim)(x)

def sample_z(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

z = layers.Lambda(sample_z)([z_mean, z_log_var])

# Decoder
decoder_input = layers.Input(shape=(latent_dim,))
x = layers.Dense(16, activation='relu')(decoder_input)
outputs = layers.Dense(input_dim)(x)
decoder = Model(decoder_input, outputs)

# Connect decoder to encoder output
decoded = decoder(z)
vae = Model(inputs, decoded)

# Step 3: Define VAE loss
reconstruction_loss = tf.reduce_mean(tf.square(inputs - decoded))  # MSE
kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
vae.add_loss(reconstruction_loss + kl_loss)
vae.compile(optimizer='adam')

# Step 4: Train the model
vae.fit(player_data, player_data, epochs=100, batch_size=32, verbose=0)

print("✅ VAE trained on synthetic sports data.")
# Generate 5 new synthetic players from latent space
for i in range(5):
    z_sample = np.random.normal(size=(1, latent_dim))
    generated_traits = decoder.predict(z_sample)
    
    # Denormalize to real-world scale (optional)
    generated_traits = generated_traits * player_data.std(axis=0) + player_data.mean(axis=0)

    print(f"\n🏃 Player {i+1}")
    print(f"Speed    : {generated_traits[0][0]:.2f}")
    print(f"Agility  : {generated_traits[0][1]:.2f}")
    print(f"Strength : {generated_traits[0][2]:.2f}")
