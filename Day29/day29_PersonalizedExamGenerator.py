# personalized_exam_generator.py

"""
This script generates personalized exam questions based on topic attention weights,
simulating how attention mechanisms prioritize important information in deep learning.
"""

import operator  # For sorting (optional, but included for clarity)

# Simulated topics and their attention weights (importance based on student performance)
topic_attention = {
    "Data Structures": 0.8,
    "DBMS": 0.6,
    "OS": 0.3,
    "Networking": 0.2,
    "AI": 0.7
}

# Sample question bank for each topic
question_bank = {
    "Data Structures": [
        "Explain different types of trees.",
        "What is a hash table? Explain with an example."
    ],
    "DBMS": [
        "What are ACID properties in DBMS?",
        "Explain normalization with examples."
    ],
    "OS": [
        "What is a deadlock? How to avoid it?",
        "Differentiate between multitasking and multiprocessing."
    ],
    "Networking": [
        "Explain the TCP/IP model.",
        "What is DNS and how does it work?"
    ],
    "AI": [
        "Explain supervised vs unsupervised learning.",
        "What is a neural network and how does it function?"
    ]
}

def generate_exam(num_questions=3):
    """
    Generates and prints a set of personalized exam questions based on attention weights.
    """
    # Sort topics based on highest attention weight
    sorted_topics = sorted(topic_attention.items(), key=operator.itemgetter(1), reverse=True)

    print("📘 Personalized Exam Questions Based on Attention Weights:\n")
    count = 0
    for topic, weight in sorted_topics:
        if count >= num_questions:
            break
        questions = question_bank.get(topic, [])
        if questions:
            print(f"➡️ {topic}: {questions[0]}")
            count += 1

if __name__ == "__main__":
    generate_exam()
