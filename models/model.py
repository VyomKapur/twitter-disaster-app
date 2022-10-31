import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def cleanText(text):
    text = re.sub(r'@[A-Za-z0-9]', '', text)
    text = re.sub(r'#[A-Za-z0-9]', '', text)
    text = re.sub(r'RT[\s]+', '', text)
    text = re.sub(r'https?:\/\/\S+', '', text)
    return text

data = pd.read_csv('./datasets/train.csv')
print(data.head())

vectorizer = CountVectorizer(stop_words='english')
x = vectorizer.fit_transform(data['text'])
pickle.dump(vectorizer, open('vec.pkl', 'wb'))

y = data['target']

model = MultinomialNB()
model.fit(x,y)

pickle.dump(model, open('model.pkl', 'wb'))

