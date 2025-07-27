# Day 26: NLP Basics (TF-IDF, Word2Vec) - ML Daily Byte Series

## 🚀 Topic
Natural Language Processing (NLP) Basics - TF-IDF and Word2Vec.

## 📚 Concepts Covered
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: A technique to convert text to numerical vectors based on word importance.
- **Word2Vec**: A neural-based technique to map words into vector space, capturing semantic meaning.

## 🧠 Real-World Advanced Example: Legal Document Classification
In this example, we:
- Load legal documents from various categories (contracts, patents, case laws, etc.)
- Convert them into TF-IDF vectors for training a Naive Bayes classifier.
- Use Word2Vec to visualize word relationships and semantic similarity within the legal domain.

## 💻 How it Works
- Preprocess legal texts (lowercase, remove punctuation, stopwords, etc.)
- Fit a TF-IDF vectorizer on the text corpus
- Train a Multinomial Naive Bayes model
- Use a pretrained Word2Vec model (like Gensim’s GoogleNews or legal-specific) for word similarity

## 📁 Files
- `legal_doc_classifier.py`: Main script for text classification and Word2Vec demo

## 📊 Output
- Accuracy of classification model on unseen legal texts
- Most similar legal terms using Word2Vec

---

### 🔧 Requirements
```bash
pip install scikit-learn gensim nltk
