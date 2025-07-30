# 🎓 Personalized Exam Generator using Attention Mechanism

This project demonstrates how **Attention Mechanisms** can be applied in education to build a **personalized exam generator**. Based on a student's weak areas, attention scores are assigned to each topic. The system then selects the most relevant questions from the question bank using these attention weights.

## 📌 Concept

The core idea is inspired by how attention works in NLP models — focusing more on important parts of the input. Similarly, here the system pays more attention to weaker topics of a student and generates questions accordingly.

- **High attention score = weaker area = more questions**
- **Low attention score = strong area = fewer questions**

## 🧠 Example

If a student performs poorly in **Data Structures (0.8)** and well in **Networking (0.2)**, the generator will focus on asking more from **Data Structures**.

## 🚀 Features

- Simulates attention scores for topics.
- Generates a set of exam questions based on priority topics.
- Easily extendable to real datasets or performance analytics.

## 🛠️ Built With

- Java
- Collections Framework (Map, List, Sorting)
- No external libraries

## 🗂️ File Structure

```
📁 PersonalizedExamGenerator
├── PersonalizedExamGenerator.java
└── README.md
```

## ▶️ How to Run

1. Copy the `PersonalizedExamGenerator.java` file into your Java IDE or editor.
2. Compile and run the program:
   ```bash
   javac PersonalizedExamGenerator.java
   java PersonalizedExamGenerator
   ```

## 💡 Output Example

```
📘 Personalized Exam Questions Based on Attention Weights:
➡️ Explain different types of trees.
➡️ What are ACID properties?
➡️ Explain supervised vs unsupervised learning.
```

## 📍 One-line Description

> Java-based simulation of attention-driven personalized exam generator that focuses on a student's weak areas using topic attention scores.

---

## 📚 Tags

`#Java` `#AIinEducation` `#AttentionMechanism` `#PersonalizedLearning` `#MachineLearningConcepts` `#DailyByte`
