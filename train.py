import pandas as pd
import logging
import time
import numpy as np
import re

import pickle

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import nltk
#from sklearn import cross_validation
#from sklearn.model_selection import GridSearchCV
#import matplotlib.pyplot as plt


from utils import ( save_brown_model,
					preprocess_text,
					csv_to_df,				
				)

# pickle protocol used to dump files. Due top
# possible errors on different python versions, using
# the safer protocol instead of highest available
# pickle_protocol = pickle.HIGHEST_PROTOCOL
pickle_protocol = 2

# nltk library
#stemmer = PorterStemmer()
#words = stopwords.words("english")
logging.basicConfig(
	filename = 'error.log',
	level=logging.INFO
)

logfile = logging.getLogger('file')

train_file = 'data/labeled_data.csv'
ref_file = 'data/reference_questions.csv'
model_file = 'data/model.pickle'

split_ratio=0.8

def load_model():
	try:
		with open(model_file, 'rb') as handle:
			classifier = pickle.load(handle)
			logfile.debug(
				"Classifier loaded in memory, size = " 
			) 
			return classifier
	except Exception as e:
		logfile.error(
			"Could not load classifier in memory" 
		) 		
	return

def train_model():
	'''
	train multiple models and compare
	accuracies, chosen classifier = 
	MultinomialNB

	rtype : None
	'''

	# create train and test sets using default ratio
	train_ques, train_codes, test_ques, test_codes = split_data(
														split_ratio
													)

	classifier = train_MultinomialNB(
					train_ques, 
					train_codes, 
					test_ques, 
					test_codes
				)

	# after training, save the classifier
	# model file as pickle on disk
	# to be loaded when project is run
	with open(model_file, 'wb') as handle:
		pickle.dump(
		classifier, 
		handle, protocol=pickle_protocol
	)

	logfile.info(
		"Classifier trained and saved as "
		+ model_file
	)

	save_brown_model()
	logfile.info(
		"Word embedding downloaded and saved as "
		+ model_file
	)	

def train_MultinomialNB(train_ques, 
						train_codes, 
						test_ques, 
						test_codes
						):
	'''
	Using library sklearn and nltk
	train a multinomial Naive Bayes
	classifier on the train data

	rtype: sklearn.naive_bayes.MultinomialNB
			object
	'''
	# called only once to tain and save
	# the model

	# Use sklean pipeline to directly feed
	# the result vector from count vectorizer
	# to tdidf vectorizer to the multinomialNB
	classifier = Pipeline([('vect', CountVectorizer()),
						('tfidf', TfidfTransformer()),
						('clf', MultinomialNB()),
						])

	# fir the training data in 
	# the classifier
	classifier = classifier.fit(
					train_ques, 
					train_codes
				)

	# predict the accuracy using 
	# the held out test set
	predicted = classifier.predict(
					test_ques
				)
	
	print(
		"MultinomialNB accuracy ="
		+ str(np.mean(predicted == test_codes))		
		)

	return classifier

def split_data(ratio):
	'''
	takes the training file, converts
	into pandas dataframe and splits
	the dataframe in train and test
	sets using default ratio 8:2

	rtype: 4 lists for x_train, y_train,
	x_test and y_test
	'''
	df = csv_to_df(train_file)

	if df.empty:
		return [],[],[],[]

	# get the column names
	cols = df.columns

	# total num of rows in dataset
	nrows = len(df)

	# split num of rows for train set
	nrows_train = int(ratio * nrows)	
	
	X = df[cols[0]].tolist() 
	y = df[cols[1]].tolist()

	x_train = X[:nrows_train]
	x_test = X[-(nrows-nrows_train):]

	y_train = y[:nrows_train]
	y_test = y[-(nrows-nrows_train):]

	return x_train, y_train, x_test, y_test

