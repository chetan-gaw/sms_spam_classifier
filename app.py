import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text=y[:]  #cloning
    y.clear()

    for i in text:
       if i not in stopwords.words('english'):
          y.append(i)

    text=y[:]  #cloning
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

tfidf = pickle.load(open('sms-vectorizer.pkl','rb'))
model = pickle.load(open('sms-model.pkl','rb'))

st.title(":blue[SMS Spam Classifier]")

input_sms = st.text_input("Enter the Message")

if st.button("Predict"):
    #1.preprocess
    transformed_sms = transform_text(input_sms)
    #2.Vectorize
    vector_input = tfidf.fit_transform([transformed_sms])
    #3.predict
    result = model.predict(vector_input)[0]
    #4.display
    if result == 1:
        st.header(":red[Spam]")

    else :
       st.header(":green[Not a Spam]")