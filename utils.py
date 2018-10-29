# RE module re provides full support for Perl-like regular
# expressions in Python
import re

import logging
import pandas as pd
# Pickle is used for serializing and de-serializing a Python 
# object structure so that it can be saved on disk and reloaded
# in memory later. Here it is used to store model files during 
# training and to load the same model files when running the project
import pickle

# NLTK based libraries
import nltk
from nltk.corpus import brown, words, stopwords
from nltk.collocations import *

# GENSIM word model
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from gensim.models import Word2Vec

from collections import Counter

# SK LEARN based libraries
#from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Using NLTK library and Brown Corpus for word embeddings
# http://www.nltk.org/howto/corpus.html
embedding_file = 'data/brown.embedding'
ref_file = 'data/reference_questions.csv'

logging.basicConfig(
	filename = 'error.log',
	level=logging.INFO
)

logfile = logging.getLogger('file')

def get_similar_words_sent(ques_text, max_sim_words, min_count):
	'''
	returns a list of similar words for every word
	in a sentence for possible string matching

	param: ques_text - string - word/phrase for which similar
			words are being searched
	param: max_sim_words - int - max no of similar words
		 	to return
	param: min_count - min number of words to return per search 
	output: list of str - all the similar words found
	'''
	if not ques_text:
		return []

	similar_words = []
	words = ques_text.split(' ')
	for word in words:
		# ignore if the word is 
		# a common stopword
		if is_stopword(word):
			continue
		similar_words.extend(
			get_similar_words_word(
				word, max_sim_words, min_count
			)
		)
	
	return similar_words

def get_ref_ques():
	'''
	returns a dict with keys
	as the question code and value
	as the reference ques text

	param: none
	output : dict
	'''
	df_ref = csv_to_df(ref_file)	
	ref_ques = dict(df_ref.apply(
								lambda row: (
									row['code'],
									row['question_text']
								),
						axis=1
					).tolist())


	return ref_ques

def calc_similarity(text1, text2):
	'''
	calculates a similarity score
	between two texts based on 
	string distance funtions
	cosine and levenshtein

	param: text1 - a string
	param: text2 - a string
	output: float value representing 
	the similarity score
	'''
	if not text1 or not text2:
		return 0

	score = 0

	# if any of the text is a single word, use
	# levenshtein distance as it is a better
	# metric to calculate similarity
	if is_single_word(text1) or is_single_word(text2):
		lev_distance = calc_levenshtein_distance(
							text1, 
							text2
						)

		if not lev_distance:
			# exact match, similarity should be highest
			score = 1
		else:		
			# since lev distance increases as similarity
			# decreases, calculate score as inversely
			# proportional	
			score = 1/lev_distance 
			
	else:
		# for strings larger than single word
		# cosine similarity is a better measure
		# of their similarities
		score = calc_cosine_distance(text1, text2)

	logfile.info("calc_similarity() :"
		+ " text1 = ("
		+ str(text1)
		+ ") text2 = ("
		+ str(text2)
		+ "), sim score = "
		+ str(score)
		)

	return score

def csv_to_df(filename, 
			delim_rows=',',
			delim_cols='|',
			encoding='iso-8859-1',
			col='question_text|code' 
			):	

	'''
	reads a csv file and converts the
	contents into pandas dataframe. Expects
	first column to have data with a 
	default delimter for rows and columns

	param: filename - csv file path to extract data from
	param: delim_rows - delimiter in csv file based on which
			row values are separated
	param: delim_cols - delimiter in column header assuming more
			than one column's data is in a column
	param: col - name of column which holds the data
	output: dataframe
	'''
	df = pd.read_csv(
		filename,
		delimiter = delim_rows,
		encoding = encoding				 
	)

	if df.empty:
		logfile.error(
			"Error reading data from file ="
			+ filename
			)
		return 

	# get the first column
	# which contains the data
	df = df[[col]]

	# split the col names based
	# on the delimiter
	cols = col.split(delim_cols)

	# split the col values based
	# on the delimiter
	df[cols] = df[col].str.split(
						'|',
						expand=True
					)

	# drop the old column
	# as the new split ones
	# contain all relevant
	# data
	df=df.drop(col, axis=1)
	
	# clean data for NA, trailing spaces and 
	# invalid characters

	df = preprocess_df(df)

	return df

def preprocess_df(
					df, 
					header=True, 
					removeSpChars=False, 
					removeTrlSpaces=True,
					dropNAValues=True,
					toLowerCase=False
					):
	'''
	Receives a pandas df and remove all
	trailing spaces, special charasters, NA
	values and None values for all columns based
	on optional parameters

	param: df - the dataframe to pre process
	param: header - boolean value for whether this df
			contains a header row or not
	param: removeSpChars - boolean remove special characters or not
	param: removeTrlSpaces - boolean remove trailing spaces or not
	param: dropNAValues - boolean remove the NA values
	param: toLowerCase - convert data to lower case
	output: df
	'''
	if df.empty:
		return

	# remove all rows which have NA and None values
	if dropNAValues:
		df = df.dropna() #(how='any',axis=0) 

	num_cols = len(df.columns)
	cols = []

	# if the frame doesnt have col names,
	# generate them
	if not header:
		col_list = ["col"+str(x) for x in range(
						num_cols
						)
					]

	else:
		col_list = list(
						df.columns.values
					)

	# for each col, perform
	# processing of data
	for col in col_list:
		if removeTrlSpaces:
			df[col] = df[col].apply(
						lambda x: x.strip()
					)

		if removeSpChars:
			df[col] = df[col].apply(
						lambda x: re.sub('-\W+','',x)
					)

		if toLowerCase:
			df[col] = df[col].apply(
							lambda x: x.lower()
						)

	return df

def preprocess_text(
					text,
					removeSpChars=False, 
					removeTrlSpaces=False,
					toLowerCase=False
				):
	'''
	clean a string word, remove spaces, punctuations,
	and other non alphanumeric characters

	input : str 
	output : str
	'''
	if not text:
		return ""

	if text == ',' or text == '.':
		return ""

	if removeTrlSpaces:
		text = text.strip()

	if removeSpChars:
		text = re.sub('\W+','',str(text))
	
	if toLowerCase:
		text = text.lower()

	return text


def is_stopword(word):
	'''
	user nltk library stop words to check if a 
	word is a stop word like 'the', 'an' etc

	input : str
	output : boolean
	'''
	if word in set(stopwords.words('english')):
		return True
	else:
		return False

def calc_levenshtein_distance(word1, word2):
	'''
	calculates the levenshtein distance (edit distance)
	between two words, good way to measure similarity 
	between one word strings

	input : str, str
	output : int 
	'''
	return nltk.edit_distance(word1, word2)

def calc_cosine_distance(*strs): 
	'''
	calculates the cosine similarity between
	words in a list. for free text as keywords, 
	cosine similarity, is a better measure 

	input : list
	output : float 
	'''
	distance=0
	try:
		vectors = [t for t in get_vectors(*strs)]
		# return only the 0,1 value as this function 
		# expects only 2 word comparisons
		# TODO extend to larger list of words
		distance = cosine_similarity(vectors)[0][1]
	except Exception as e:
		# if there is an exception thrown
		# from the library, ignore it, since
		# the cosine similarity is optional
		# in finding related topics
		pass
	return distance

def get_vectors(*strs):
	'''
	calculates the numeric expression (nparray)
	for string using sklearn library for
	feature_extraction

	input : list of str
	output : matrix of floats
	'''
	text = [t for t in strs]
	vectorizer = CountVectorizer(text)
	vectorizer.fit(text)
	return vectorizer.transform(text).toarray()


def get_similar_words_word(word, topN, min_count):
	'''
	using gensim model word2Vev, find the closest
	topN words for a given word

	input : str, int, int
	output : list (str))	
	'''
	most_similar = []
	try:
		model = gensim.models.Word2Vec.load(
					embedding_file
				)

		most_similar = model.most_similar(
						word, 
						topn=topN
					)
		logfile.info(
			"get_most_similar() for word = ("
			+ str(word)
			+ "), similar words generated = "
			+ str(most_similar)
			)

	except Exception as e:
		# pass this exception since there may be words 
		# like proper nouns for which the gensim model
		# will throw error
		pass

	result = []
	# get the word from the tuple
	# of word and probability
	for each in most_similar:
		result.append(each[0])

	# return list of most similar words
	return result

def save_brown_model():
    '''
    nltk brown corpus is downloaded and saved
    locally for word lookup later, called once
    during training of data

	input : none
	output : none
    '''
    sentences = brown.sents()
    model = gensim.models.Word2Vec(sentences, min_count=1)
    model.save('data/brown.embedding')

def spell_check(word):
	'''
	if word is a misspelling, try to find closest
	match of a valid word from nltk using its 
	jaccard distance function

	input : str
	output : str	
    '''
	correct_spellings = words.words()
	spelled = ""
	c = [i for i in correct_spellings if i[0]=='c']

	one = [(nltk.jaccard_distance(set(nltk.ngrams(word, n=2)), \
				set(nltk.ngrams(a, n=2))), a) for a in c]

	# sort them to ascending order so shortest 
	# distance (closer matches) are picked first
	# extract only the word 
	distance, word = sorted(one)[0]

	if isinstance(word, str):
		spelled = preprocess_text(word)

	return spelled

def is_single_word(word):
	'''
	check if a single word or phrase

	input : str
	output : boolean	
	'''
	if len(word.split()) == 1:
		return True
	else:
		return False
