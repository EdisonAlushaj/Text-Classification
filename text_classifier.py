import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

try:
    stopwords.words('english')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
    print("Download complete.")


print("\n--- Loading Dataset ---")

categories = ['rec.sport.hockey', 'sci.med', 'soc.religion.christian', 'comp.graphics']
dataset = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42, remove=('headers', 'footers', 'quotes'))

df = pd.DataFrame({'text': dataset.data, 'category': [dataset.target_names[i] for i in dataset.target]})

print("Dataset loaded successfully.")
print(f"Number of samples: {len(df)}")
print("First 5 rows of the dataset:")
print(df.head())

print("\n--- Performing Exploratory Analysis ---")

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='category')
plt.title('Number of Samples per Category')
plt.ylabel('Count')
plt.xlabel('Category')
plt.show()

print("\n--- Preprocessing Text Data ---")

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Cleans text data by:
    1. Making it lowercase.
    2. Removing punctuation and numbers.
    3. Removing common English 'stopwords'.
    """

    text = text.lower()

    text = re.sub(r'[^a-z\s]', '', text)

    words = text.split()

    words = [word for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(words)

df['processed_text'] = df['text'].apply(preprocess_text)

print("Text preprocessing complete.")
print("Dataset after adding processed text column:")
print(df.head())


print("\n--- Splitting Data and Vectorizing ---")

X = df['processed_text']
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Data split and vectorized.")


print("\n--- Training and Evaluating Model ---")

model = MultinomialNB()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.xlabel('Predicted Category')
plt.ylabel('Actual Category')
plt.title('Confusion Matrix')
plt.show()


print("\n--- Prediction Function ---")

def predict_category(sentence):
    """
    Takes a sentence, preprocesses it, vectorizes it, and predicts its category.
    """
    processed_sentence = preprocess_text(sentence)
    sentence_vec = vectorizer.transform([processed_sentence])
    prediction = model.predict(sentence_vec)
    return prediction[0]

test_sentence_1 = "The goalie made a great save during the hockey game."
test_sentence_2 = "Doctors and scientists are researching new cures for diseases."

print(f"Sentence: '{test_sentence_1}'")
print(f"Predicted Category: {predict_category(test_sentence_1)}")
print("-" * 20)
print(f"Sentence: '{test_sentence_2}'")
print(f"Predicted Category: {predict_category(test_sentence_2)}")