import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

data = pd.read_csv("spam.csv", encoding="latin-1")

X = data['v2']
y = data['v1']

cv = CountVectorizer()
X = cv.fit_transform(X)

model = MultinomialNB()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(cv, open("vectorizer.pkl", "wb"))

print("Model Trained Successfully!")