# 🧠 Transfer Learning for Pneumonia Detection using Chest X-rays

This project demonstrates the power of Transfer Learning by using a pre-trained DenseNet121 model to detect pneumonia from a small dataset of chest X-ray images. Instead of training from scratch, the model leverages learned features from ImageNet and adapts them to a specialized medical task.

---

## 📌 Problem Statement

Detecting pneumonia using chest X-rays is a critical task in healthcare. However, training an accurate deep learning model from scratch requires a large number of labeled images, which are hard to obtain in medical settings.

---

## 💡 Solution: Transfer Learning

By applying transfer learning:
- We use DenseNet121, a CNN pre-trained on ImageNet.
- We replace the final classification layer to output 2 classes: `Pneumonia` and `Normal`.
- We fine-tune only the last few layers on our medical dataset.

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Pretrained DenseNet121
- Matplotlib & NumPy for visualization
- Chest X-ray dataset (e.g., from [Kaggle NIH Chest X-ray](https://www.kaggle.com/datasets/nih-chest-xrays/data))

---

## 📊 Results

- Achieved high accuracy on validation set using a relatively small dataset
- Significantly reduced training time and computational cost
- Shows the effectiveness of transfer learning in medical imaging

---

## 📁 Folder Structure

