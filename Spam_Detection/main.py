

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

#  1: Collect Email Data
# Assuming you have a CSV file named 'emails.csv' with two columns: 'text' (email content) and 'label' (spam or ham)
data = pd.read_csv("C://Users//user//Downloads//archive (2)//spam.csv", encoding='ISO-8859-1')


#  2: Preprocess the Data
# You may need to clean

# Step 3: Feature Extraction

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['v2'])

y = data['v1']

#  4: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  5: Choose a Machine Learning Model
# use Naive Bayes classifier
model = MultinomialNB()

#  6: Train the Model
model.fit(X_train, y_train)

#  7: Evaluate the Model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, predictions))
engine.say(classification_report(y_test, predictions))
engine.runAndWait()
#  8: Fine-Tune the Model (Optional)
# You can try different algorithms to improve performance

#  9: Deploy the Model
#  deploy it for making predictions on new email data


