import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('datasets/bbc-text.csv')

# Two ways to convert text into numeric
count_vectorizer = CountVectorizer()
x_train_count = count_vectorizer.fit_transform(data['text'])

tfidf_vectorizer = TfidfVectorizer()
x_train_tfidf = tfidf_vectorizer.fit_transform(data['text'])

# Train model
Y = data['category']

x_train, x_test, y_train, y_test = train_test_split(x_train_tfidf, Y, test_size=0.2)

classifier = DecisionTreeClassifier(max_depth=10)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print("Accuracy score : ", accuracy_score(y_test, y_pred))

