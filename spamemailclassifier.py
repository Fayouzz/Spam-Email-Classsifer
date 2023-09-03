import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset (X: email text, y: labels)
df = pd.read_csv('spam.csv')
# Preprocess the data (remove HTML, tokenize, etc.)
# 'Text' is the column containing email text, and 'Category' is the column containing labels
X = df['Message']  # Replace 'Text' with the actual column name in your dataset
y = df['Category']  # Replace 'Category' with the actual column name in your dataset

# 'ham' and 'spam' labels can be encoded as 0 and 1, respectively
y = y.map({'ham': 0, 'spam': 1})
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data into TF-IDF features
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Choose and train a machine learning model (e.g., Multinomial Naive Bayes)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))
