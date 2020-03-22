import pandas as pd 
from nltk.corpus import stopwords
import string
import pickle
import warnings

warnings.filterwarnings('ignore')

def text_process(mess):
	'''
	#1. Remove punctuations
	2. Remove stop words
	3. Return list of clean text words
	'''
	no_punc = [char for char in mess if char not in string.punctuation]
	no_punc = ''.join(no_punc)
	return [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]

spam_classifier = pickle.load(open('models\\spam_classifier.dat', 'rb'))
vectorizer = pickle.load(open('models\\vectorizer.dat', 'rb'))
tfidf = pickle.load(open('models\\tfidf.dat', 'rb'))

text_input = input("Enter your message: ")

text_input = pd.Series(text_input)

vec = vectorizer.transform(text_input)

preprocessed_text = tfidf.transform(vec)

predictions = spam_classifier.predict(preprocessed_text)

print("\nThe message is: {}\n".format(predictions[0]))