import pandas as pd
import nltk
import re
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

messages = pd.read_csv('data/data_spamclassifier/sms+spam+collection/SMSSpamCollection', sep='\t', 
                       names=["label","message"])

data = list(messages['message'])

stemmer = PorterStemmer()

corpus = []

#Data Cleaning and Preprocessing
for i in range(len(data)):
    review = re.sub('[^a-zA-z]', ' ', data[i])
    review = review.lower()
    review = review.split()


    review = [stemmer.stem(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

#Creating the Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])

y = y.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20, random_state=0)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB()

spam_detect_model.fit(X_train,y_train)

y_pred = spam_detect_model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix
confusion_m = confusion_matrix(y_test,y_pred)
print('Confusion_Matirx :\n', confusion_m)
print('Accuracy :', accuracy_score(y_test,y_pred))
