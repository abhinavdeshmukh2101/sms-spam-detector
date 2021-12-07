import numpy as np
import pandas as pd
import nltk
import string
import pickle
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
nltk.download('punkt')
nltk.download('stopwords')

encoder = LabelEncoder()
ps = PorterStemmer()
svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
data_frame = pd.read_csv('spam.csv')

data_frame.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)

data_frame.rename(columns={'v1':'target','v2':'text'},inplace=True)

#using encoder to change ham value to 0 and spam value to 1(assigned automatically)
data_frame['target'] = encoder.fit_transform(data_frame['target'])  # spam -1 and ham 0

# data_frame.isnull().sum()

# check for duplicate values and removes them(except the first duplicate)
# data_frame.duplicated().sum()
data_frame = data_frame.drop_duplicates(keep='first')

# data_frame['target'].value_counts()

#num of character
data_frame['num_characters'] = data_frame['text'].apply(len)

# num of words
data_frame['num_words'] = data_frame['text'].apply(lambda x:len(nltk.word_tokenize(x)))

#num of sentence
data_frame['num_sentences'] = data_frame['text'].apply(lambda x:len(nltk.sent_tokenize(x)))

def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)
  
  y = []
  for i in text:
      if i.isalnum():
          y.append(i)
  
  text = y[:]
  y.clear()
  
  for i in text:
      if i not in stopwords.words('english') and i not in string.punctuation:
          y.append(i)
          
  text = y[:]
  y.clear()
  
  for i in text:
      y.append(ps.stem(i))
  
          
  return " ".join(y)

#pre-processing of text messages.
data_frame['transformed_text'] = data_frame['text'].apply(transform_text)

#
spam_corpus = []
for msg in data_frame[data_frame['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)

ham_corpus = []
for msg in data_frame[data_frame['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(data_frame['transformed_text']).toarray()
y = data_frame['target'].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

mnb.fit(X_train,y_train)
y_pred2 = mnb.predict(X_test)
# print(accuracy_score(y_test,y_pred2))
# print(confusion_matrix(y_test,y_pred2))
# print(precision_score(y_test,y_pred2))

voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting='soft')
y_pred = voting.predict(X_test)

estimators=[('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator=RandomForestClassifier()

clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)


pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))
