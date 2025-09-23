import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Load CSV
df = pd.read_csv("./data/values_emotions_joy.csv")

# Features (instructions) and labels (outputs)
X = df["instruction"]
y = df["output"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline: TF-IDF + Logistic Regression
clf = Pipeline(
    [
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("logreg", LogisticRegression(max_iter=200)),
    ]
)

# Train
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Example usage
example = ["you get a promotion at work"]
print("Prediction:", clf.predict(example)[0])
