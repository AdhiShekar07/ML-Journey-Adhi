from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample dataset
emails = [
    "Win a brand new car just by clicking this link!",
    "Lowest price guaranteed on your loan today",
    "Hey, are we still meeting at 5 pm?",
    "Your Amazon order has been shipped",
    "You have WON a lottery of $1,000,000!",
    "Reminder: your meeting is at 2:00 PM today",
    "Get cheap pills now! No prescription needed",
    "Lunch at my place tomorrow?",
    "URGENT! Confirm your bank account now"
]

labels = [1, 1, 0, 0, 1, 0, 1, 0, 1]  # 1 = spam, 0 = not spam

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict on a new message
sample = ["Congratulations! You've won a free iPhone!"]
sample_vector = vectorizer.transform(sample)
prediction = model.predict(sample_vector)

print("Prediction for sample message:", "Spam" if prediction[0] == 1 else "Not Spam")
