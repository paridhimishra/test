import json
import logging
import collections
import numpy as np
from train import load_model

from utils import (calc_similarity, 
					get_ref_ques,
					preprocess_text,
					get_similar_words_sent,
					spell_check
				)

logging.basicConfig(
	filename = 'error.log',
	level=logging.DEBUG
)

logfile = logging.getLogger('file')

# weight given to the classifier
# probability to contribute to 
# relevance score
weight_prob = 0.5
# weight given to the similarity
# score to contribute to 
# relevance score
weight_sim = 0.5
# min similarity threshold
# below which the results
# are discarded as less relevant
min_sim_score = 0.05

# max number of similar words
# to find per word in a text 
max_sim_words = 2

# min number of words to 
# return per similar word search
min_count = 1

def match_and_rank(ques_text, topn):
	'''
	given a ques text, ranks the 
	matched reference questions
	using multi classification
	and string similarity techniques

	rtype: a dict with topn number of 
	items, each item is a dict its rank
	as its key and set of 3 values 
	(reference question text, 
	reference question code,
	and relevance score)
	as its values

	'''
	if not ques_text or int(topn)<1:
		return json.dumps({})

	if ques_text.isdigit():
		return json.dumps({})

	result = {}
	sorted_result = {}
	ranked_result = {}

	extend_search = False
	sim_words = []

	ref_ques_dict = get_ref_ques()

	ques_text = similarity_check(ques_text, ref_ques_dict)
	sim_scores =  get_sim_ref_ques(ques_text)


	# if no similarity match found
	# with any of reference questions
	# dont fall back on the classifier
	# predictions else very irrelavent 
	# results served using class alone
		
	# get the class predictions from
	# the classifier
	class_preds = classify(ques_text)

	for class_pred in class_preds:

		# predicted code from classifier
		code = class_pred[0]
		# probability of predicted class
		try:
			class_prob = float(class_pred[1])
		except Exception as e:
			class_prob = 0.0

		ref_ques = ref_ques_dict[code]

		# calculate relevance score
		# for totally dissimilar strings, class prob
		# is generated, so it might be misleading
		rel_score =  (
						(weight_prob*class_prob) 
						+(weight_sim*sim_scores[code])
					)

		# temp dict to store each
		# class details
		tmp = {}
		tmp['reference_question'] = ref_ques
		tmp['predicted_code'] = code
		tmp['relevance_score'] = rel_score

		if  sum(sim_scores.values()) > min_sim_score:
			result[rel_score] = tmp


	# Sort the result in decreasing order
	# of relevance score
	sorted_result = reversed(sorted(result))

	for count, val in enumerate(sorted_result):
		# return only topn results
		if count == int(topn):
			break

		ranked_result[count+1] = result[val]

	logfile.info(
		"match_and_rank() : ques_text = (" 
		+ str(ques_text)
		+ ")"
		+ " result = "
		+ str(ranked_result)
	)

	return (ranked_result)

def similarity_check(ques_text, ref_ques_dict):
	'''
	for a given ques text, checks if its sum
	of similarity scores with all reference
	questions is less than min score, if yes
	takes each word from the ques and finds
	similar words, also tries spell check
	and returns a list of possible words
	similar to original ques text

	rtype: list of str
	'''
	if not ques_text:
		return ''
	# remove trailing spaces, spl characters
	ques_text = preprocess_text(ques_text)

	sim_words = []
	# get the similarity scores from
	# the string matching algorithms
	sim_scores = get_sim_ref_ques(ques_text)
	
	if  sum(sim_scores.values()) <= min_sim_score:
		# if the similarity score is very low
		# lower than minimum accepted, try
		# finding similar words as a fallback
		# option #1
		sim_words = get_similar_words_sent(
						ques_text, 
						max_sim_words, 
						min_count
					)

		if not sim_words:
			# if there are no similar words
			# there is a possibility that 
			# this word has been misspelled
			# try to do a spell check
			spelled = spell_check(ques_text)
			sim_words = [spelled]

	if sim_words:
		# if either the bigrams from gensim
		# word vec or spell check generated 
		# similar words, add to the original
		# question text
		ques_text = str(ques_text).replace('"', '')
		new_text = str(" ".join(sim_words))
		ques_text = ques_text+ " "+ new_text

	return ques_text

def classify(ques_text):
	'''
	uses a pre trained model 
	to predict the class (code) of the ques text

	rtype: a reverse sorted dict with class 
	probability and the question code for the 
	given ques text
	'''
	if not ques_text:
		return {}

	classifier = load_model()

	if not classifier:
		return {}

	cl_codes = classifier.classes_
	cl_probs = classifier.predict_proba(
					[ques_text]
				)

	# the classifier returns unpredictable string
	# which needs to be cleaned before converting
	# into list
	cl_probs = str(cl_probs).replace(
					'[',''
				).replace(
					']',''
					).replace(
						'  ',' '
						).split(' ')

	result = list(zip(cl_codes, cl_probs))
			
	logfile.info(
		"classify() : ques_text = "
		+ str(ques_text)
		+ " Class probability matrix = "
		+ str(result)
		)

	return result

def get_sim_ref_ques(ques_text):
	'''
	takes the ques text and finds
	the similarity score against 
	each of the ref questions

	rtype: a dict with key as the 
	ref question code and the similarity
	score as the value
	'''
	result = {}

	# get the reference questions
	# stored along with their codes
	ref_ques_dict = get_ref_ques()

	if not ref_ques_dict:
		return {}

	# for each reference ques and given ques text
	for ref_code, ref_ques in ref_ques_dict.items():

		# calculate the similarity between
		# the two string based as a num
		# between 0 and 1
		sim_score =  calc_similarity(
						ques_text,
						ref_ques
					)

		result[ref_code] = sim_score

	logfile.info(
		"match_strings() : ques_text = "
		+ str(ques_text)
		+ " Similarity score matrix = "
		+ str(result)
		)

	return result
