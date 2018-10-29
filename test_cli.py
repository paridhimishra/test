import unittest
import cli
import pickle

from train import load_model, split_data
from utils import calc_cosine_distance, calc_levenshtein_distance, get_ref_ques
from text_match import classify, get_sim_ref_ques

class BaseCase(unittest.TestCase):

	def test_classify_empty_text(self):
		text = ""
		result = classify(text)

		self.assertEqual(result, {})

	def test_classify_valid_text(self):
		text = "The information I need to do my job effectively is readily available"
		result = classify(text)

		self.assertEqual(result[0][0], 'ALI.5')
		self.assertEqual(result[0][1], '4.63372393e-03')

	def test_classify_valid_text_with_sp_chars(self):
		text = "The ~~ information I need to do my job ++ effectively is readily available"
		result = classify(text)

		self.assertEqual(result[0][0], 'ALI.5')
		self.assertEqual(result[0][1], '4.63372393e-03')

	def test_get_ref_ques(self):

		ref_ques = get_ref_ques()

		self.assertEqual(len(ref_ques),4)
		self.assertEqual(ref_ques['ALI.5'], "I know what I need to do to be successful in my role")
		self.assertEqual(ref_ques['TEA.2'], "We hold ourselves and our team members accountable for results")
		self.assertEqual(ref_ques['ENA.3'], "The information I need to do my job effectively is readily available")
		self.assertEqual(ref_ques['INN.2'], "We are encouraged to be innovative even though some of our initiatives may not succeed")

	def test_get_sim_ref_ques(self):
		text = "The information I need to do my job effectively is sometimes not there"
		ref_ques = get_ref_ques()
		sim_ref_ques = get_sim_ref_ques(text)

		self.assertEqual(round(sim_ref_ques['ALI.5'],4), 0.4003)
		self.assertEqual(round(sim_ref_ques['TEA.2'],4), 0.0)
		self.assertEqual(round(sim_ref_ques['ENA.3'],4), 0.7833)
		self.assertEqual(round(sim_ref_ques['INN.2'],4), 0.1491)

	def test_load_model(self):

		classifier = load_model()
		self.assertNotEqual(classifier, {})

	def test_exists_word_embedding(self):

		model_file = 'data/brown.embedding'
		with open(model_file, 'rb') as handle:
			embedding = pickle.load(handle)

		self.assertNotEqual(embedding, {})

	def test_exists_model(self):

		model_file = 'data/model.pickle'
		with open(model_file, 'rb') as handle:
			model = pickle.load(handle)

		self.assertNotEqual(model, {})

	def test_split_data(self):

		input_file = 'data/test_questions.txt'
		x_train, y_train, x_test, y_test = split_data(0.8)
		self.assertNotEqual(x_train, {})
		self.assertNotEqual(y_train, {})
		self.assertNotEqual(x_test, {})
		self.assertNotEqual(y_test, {})

	def test_calc_cosine_distance(self):

		text1 = "The information I need to do my job effectively is readily available"
		text2 = "Is available when I really need them"

		cosine11 = round(calc_cosine_distance(text1, text1),4)
		cosine12 = round(calc_cosine_distance(text1, text2),4)

		self.assertNotEqual(cosine11, 0)
		self.assertEqual(cosine12, 0.3693)

	def test_calc_levenshtein_distance(self):

		text1 = "Employee"
		text2 = "Employed"

		leven11 = calc_levenshtein_distance(text1, text1)
		leven12 = calc_levenshtein_distance(text1, text2)

		self.assertEqual(leven11,0)
		self.assertEqual(leven12,1)


def main():
	unittest.main()

if __name__ == "__main__":
	main()