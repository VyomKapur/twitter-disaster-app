import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv('./datasets/train.csv')
print(data.head())

vectorizer = CountVectorizer(stop_words='english')
x = vectorizer.fit_transform(data['text'])
pickle.dump(vectorizer, open('vec.pkl', 'wb'))

y = data['target']

model = MultinomialNB()
model.fit(x,y)

pickle.dump(model, open('model.pkl', 'wb'))

