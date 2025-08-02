# 🏋️ VAE Sports Player Generator

A **Variational Autoencoder (VAE)** that learns from synthetic sports player data (speed, agility, strength) to generate new, realistic player profiles — just like a smart talent scout simulating potential stars.

---

## 📌 What is an Autoencoder?

An **Autoencoder** is a type of neural network that:
- Compresses input into a smaller latent space (**Encoder**)
- Reconstructs it back to original form (**Decoder**)

```text
Input ➡️ Encoder ➡️ Latent Space ➡️ Decoder ➡️ Output
```

---

## 🧠 Real-Life Analogy in Sports

Imagine you're a **sports talent scout**. You observe and rate players on key traits like:
- Speed
- Agility
- Strength

A **VAE** is like a smart system that:
- Learns how real players are distributed
- Samples new player profiles that are statistically similar
- Helps simulate potential candidates

---

## 🚀 What is a Variational Autoencoder (VAE)?

A **VAE** is an enhanced autoencoder that:
- Learns a **distribution** (mean & variance) in latent space
- Uses **random sampling** to generate realistic and smooth outputs

```text
Encoder ➡️ μ (mean), σ² (variance) ➡️ z (latent vector) ➡️ Decoder ➡️ Reconstructed Player
```

---

## 📊 Dataset Description

We generate **synthetic player data** with the following features:
- Speed ~ N(70, 10)
- Agility ~ N(65, 12)
- Strength ~ N(75, 8)

Each player is normalized and passed through the VAE model.

---

## 🛠️ Tools & Libraries Used

- Python 🐍
- TensorFlow / Keras 🧠
- NumPy 📈
- Matplotlib 📊

---

## 💻 How It Works (Key Code Steps)

1. **Data Simulation**  
   Generate 1,000 synthetic sports players using normal distributions.

2. **Model Architecture**  
   - Encoder: Compress 3D input to 2D latent space  
   - Latent Space: Learn μ and σ² for distribution  
   - Decoder: Reconstruct original 3D traits

3. **Loss Function**  
   - **Reconstruction Loss** (MSE)  
   - **KL Divergence** for smooth latent space

4. **Training**  
   Fit on synthetic data for 100 epochs.

5. **Generate New Players**  
   Sample from latent space to create new, never-before-seen players.

---

## 🎯 Sample Output

```text
🏃 Player 1
Speed    : 74.56
Agility  : 68.34
Strength : 79.23

🏃 Player 2
Speed    : 66.13
Agility  : 63.71
Strength : 74.91
```

---

## 📂 File Structure

```text
📁 VAE-Sports-Player-Generator
│
├── vae_sports_player_generator.py   # Main VAE implementation
├── README.md                        # Project documentation
```

---

## 📌 Real-World Applications

- AI-based player scouting systems
- Game character generation
- Sports performance simulation
- Player profile interpolation

---

## ✅ Concepts Covered

```text
#Autoencoder
#VariationalAutoencoder
#LatentSpace
#SportsAnalytics
#PlayerSimulation
#VAEFromScratch
#KerasVAE
#AnomalyDetection
#DataGeneration
#MachineLearningProject
```

---

## 🧠 Author

**Aditya S. (Adhi)**  
*AI & ML Enthusiast | Sports + Tech Lover | Folk Sage Creator*

---

Feel free to ⭐ star the repo if you find it useful!
