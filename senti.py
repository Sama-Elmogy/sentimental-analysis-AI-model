import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import joblib
import numpy as np
import sys
import os

# Suppress NLTK download message
nltk_data_path = os.path.join(os.path.expanduser('~'), 'AppData', 'Roaming', 'nltk_data')

if not os.path.exists(os.path.join(nltk_data_path, 'corpora', 'stopwords')):
    sys.stdout = open(os.devnull, 'w')
    nltk.download('stopwords')
    sys.stdout = sys._stdout_

data = pd.read_csv('E:\IMDB Dataset.csv')

data.rename(columns={'review': 'content'}, inplace=True)
data.dropna(subset=['content'], inplace=True)

data['cleaned_content'] = data['content'].apply(lambda x: re.sub(r'<.*?>', '', x))
data['cleaned_content'] = data['cleaned_content'].str.lower().str.replace(r'\W', ' ', regex=True)

X = data['cleaned_content']
y = data['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'))

X_train_tfidf = vectorizer.fit_transform(X_train)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
# Save the trained model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')  # Save the model
joblib.dump(vectorizer, 'vectorizer.pkl')  # Save the vectorizer

def predict_sentiment(content):
    content_cleaned = re.sub(r'<.*?>', '', content).lower().replace(r'\W', ' ')
    content_tfidf = vectorizer.transform([content_cleaned])
    sentiment = model.predict(content_tfidf)[0]
    confidence = np.max(model.predict_proba(content_tfidf))
    return sentiment, confidence


print("Welcome to the Sentiment Analysis Tool!")
print("Type your movie review to analyze its sentiment, or type 'exit' to quit.\n")

while True:
    user_sentence = input("Enter your review: ")

    if user_sentence.lower() == 'exit':
        print("Thank you for using the Sentiment Analysis Tool. Goodbye!")
        break

    sentiment, confidence = predict_sentiment(user_sentence)
    print(f"Sentiment: {sentiment.upper()}")
    print(f"Confidence Score: {confidence:.2f}")
    print("--------------------------------------------------")