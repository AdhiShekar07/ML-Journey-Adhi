import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# 1. Load Dataset
url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/heart.csv"
df = pd.read_csv(url)

# 2. Split data into features and target
X = df.drop('target', axis=1)  # All columns except target
y = df['target']               # 1 = risk of heart attack, 0 = no risk

# 3. Split into training and testing sets (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Train the model using Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Predict the results
y_pred = model.predict(X_test)

# 7. Evaluate using Confusion Matrix and Classification Report
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
