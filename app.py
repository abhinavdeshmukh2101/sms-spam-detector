#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

#Initialize the flask App
app = Flask(__name__)

tfidf = pickle.load(open('vectorizer.pkl','rb'))  # pickle.load loads the given file in binary mode.
model = pickle.load(open('model.pkl','rb'))       # rb is flag used to open the binary file in read mode.

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

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    #getting input from index.html form
    int_features = request.form.get("message")

    transformed_sms = transform_text(int_features)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        return render_template('index.html', prediction_text='Spam')
    else:
        return render_template('index.html', prediction_text='Not Spam')

    

if __name__ == "__main__":
    app.run(debug=True)