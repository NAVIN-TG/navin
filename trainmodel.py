import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# --- STEP 1: LOAD DATASET ---
# For demonstration, we create a small CSV. In a real scenario, use pd.read_csv('news.csv')

# --- STEP 1: LOAD DATASET (KAGGLE DATASET) ---

fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

fake["label"] = 0
real["label"] = 1

df = pd.concat([fake, real], axis=0)

df = df.sample(frac=1).reset_index(drop=True)

print("Dataset Shape:", df.shape)




# --- STEP 2: PREPROCESSING ---
def clean_text(text):
    """
    Cleans text by removing punctuation, numbers, and stopwords, 
    and converting to lowercase.
    """
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\\W', ' ', text) 
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)    
    return text

df['text'] = df['text'].apply(clean_text)

# --- STEP 3: SPLIT DATA ---
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- STEP 4: TF-IDF VECTORIZATION ---
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# --- STEP 5: TRAIN LOGISTIC REGRESSION ---
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# --- STEP 6: EVALUATION ---
predictions = model.predict(X_test_tfidf)
print(f"**Accuracy:** {accuracy_score(y_test, predictions)}")
print("\n**Classification Report:**")
print(classification_report(y_test, predictions, target_names=['Fake', 'Real']))

# Display Confusion Matrix
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'Real'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

import pickle

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model and Vectorizer Saved Successfully")


# --- STEP 7: CUSTOM PREDICTION ---
def predict_news(news_text):
    """
    Takes custom text input and predicts if it is Fake or Real.
    """
    cleaned_input = clean_text(news_text)
    vectorized_input = vectorizer.transform([cleaned_input])
    prediction = model.predict(vectorized_input)
    
    if prediction[0] == 0:
        return "Prediction: 🚩 FAKE NEWS"
    else:
        return "Prediction: ✅ REAL NEWS"
    
    print(df.shape)



# Example Usage
user_input = "NASA discovers water on the moon's surface."
print(f"\n**Custom News Test:** '{user_input}'")
print(predict_news(user_input))
