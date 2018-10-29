# cli.py
import click
import json

from text_match import match_and_rank
from train import train_model
import warnings
warnings.filterwarnings("ignore")

@click.command()
@click.argument('input_file')
@click.option('--topn', '-n')

def main(input_file, topn):

	'''
	called from cli with a question text
	and optional parameter topn to return 
	topn matched questions

	rtype: dict of topn results with the 
	question text, the predicted code 
	and the relevance score
	'''
	# train_model is run once to generate and store
	# the model files under data directory
	#train_model()

	# if optional param topn
	# not entered, default is 2

	if not topn:
		topn = 2

	if not input_file:
		return

	with open(input_file, 'r') as file:
		for line in file:

			# get the result as a dict, returns
			# empty dict in case of errors like
			# empty text or no matching
			# questions
			result = match_and_rank(str(line), topn)

			# print the result on stdout
			click.echo("{}".format(result))
			'''
			click.echo("{}".format(
				" The question entered  = (" 
				+ str(line)
				+ ")")
			)

			click.echo("{}".format(
				" Returning topN = " 
				+ str(topn)
				+ " matched reference questions")
			)
			'''
if __name__ == "__main__":
    main()
    