import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression

# Sample Legal Dataset (Mock)
docs = [
    "The parties agree to the terms of the license fee and duration of the contract.",
    "The plaintiff filed a claim against the defendant for breach of duty.",
    "This invention relates to a novel mechanical apparatus.",
    "The court ruled in favor of the claimant and awarded damages.",
    "Patent rights are granted to the applicant for their innovation.",
    "The agreement includes clauses about arbitration and dispute resolution.",
    "The judge issued a judgement after reviewing the evidence.",
    "The patent covers the technical details of the new invention."
]

labels = [
    "Contract",
    "Court Ruling",
    "Patent",
    "Court Ruling",
    "Patent",
    "Contract",
    "Court Ruling",
    "Patent"
]

# TF-IDF + Naive Bayes Classifier Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', MultinomialNB())
])

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(docs, labels, test_size=0.3, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Predict and Evaluate
y_pred = pipeline.predict(X_test)
print("=== TF-IDF + Naive Bayes Results ===")
print(classification_report(y_test, y_pred))


# Optional: Word2Vec for semantic similarity (not used in classification here)
tokenized_docs = [doc.lower().split() for doc in docs]
w2v_model = Word2Vec(sentences=tokenized_docs, vector_size=50, window=3, min_count=1, workers=2)

# Example: Find similar words to 'judgement'
print("\n=== Word2Vec Similar Words to 'judgement' ===")
print(w2v_model.wv.most_similar('judgement'))
