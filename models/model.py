import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('./datasets/train.csv')
print(data.head())

vectorizer = CountVectorizer(stop_words='english')
x = vectorizer.fit_transform(data['text'])
pickle.dump(vectorizer, open('./models/vec.pkl', 'wb'))

y = data['target']

model = MultinomialNB()
model.fit(x,y)

model_g = GaussianNB()
model_g.fit(x.toarray(),y)

pickle.dump(model, open('./models/model.pkl', 'wb'))
pickle.dump(model_g, open('./models/model_g.pkl', 'wb'))

