import nltk
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import string
import pickle

sns.set_style("whitegrid")

messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]
print(len(messages))

for mess_no, message in enumerate(messages[:10]):
	print(mess_no, message)
	print('\n')

messages = pd.read_csv("smsspamcollection\\SMSSpamCollection", sep='\t', names = ["label", "messages"])

print(messages.head())
print(messages.describe())

print(messages.groupby('label').describe())
messages['length'] = messages['messages'].apply(len)
print(messages.head())

messages['length'].plot.hist(bins = 70)
plt.show()

print(messages[messages['length'] == 910].iloc[0])

messages.hist(column = 'length', by = 'label', bins = 60, figsize = (12,4))
plt.show()

def text_process(mess):
	'''
	1. Remove punctuations
	2. Remove stop words
	3. Return list of clean text words
	'''
	no_punc = [char for char in mess if char not in string.punctuation]
	no_punc = ''.join(no_punc)
	return [word for word in no_punc.split() if word.lower() not in stopwords.words('english')]

print(messages['messages'].head(5).apply(text_process))	

bow_trans = CountVectorizer(analyzer = text_process).fit(messages['messages'])

print(len(bow_trans.vocabulary_))

mes = messages['messages'][3]
print(mes)

bow = bow_trans.transform([mes])
print(bow)
print(bow.shape)

messages_bow = bow_trans.transform(messages['messages'])

print("Shape of Sparse Matrix: ", messages_bow)
print(messages_bow.nnz)

sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('Sparsity: {}'.format(sparsity))

tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf = tfidf_transformer.transform(bow)
print(tfidf)

print(tfidf_transformer.idf_[bow_trans.vocabulary_['hello']])

mess_tfidf = tfidf_transformer.transform(messages_bow)

# actual prediction 
msg_train, msg_test, label_train, label_test = train_test_split(messages['messages'], messages['label'], test_size = 0.3)

pipeline = Pipeline([
	('bag_of_words', CountVectorizer(analyzer = text_process)), 
	('tfidf', TfidfTransformer()), 
	('classifier', MultinomialNB())])

pipeline.fit(msg_train, label_train)
predictions = pipeline.predict(msg_test)

print(classification_report(label_test, predictions))

print(confusion_matrix(label_test, predictions))

with open('models\\vectorizer.dat', 'wb') as f:
    pickle.dump(bow_trans, f)

with open('models\\tfidf.dat', 'wb') as f:
    pickle.dump(tfidf_transformer, f)

with open('models\\spam_classifier.dat', 'wb') as f:
    pickle.dump(spam_detect_model, f)        