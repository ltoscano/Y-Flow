
import numpy as np
from nltk.tokenize import RegexpTokenizer
import itertools




# query is .txt file of query terms
# text is .txt file of text
# distance searches for the words in query in text, then gives a score based on the total distance between
# the first and last word in a path.  All paths include all words in q in order
# ALL terms in query are ASSUMED to be present somewhere in text

def distance(query, text):

	# Tokenize text and query. Ignore punctuation and make lowercase
	tokenizer = RegexpTokenizer(r'\w+')
	q = tokenizer.tokenize(query)
	txt = tokenizer.tokenize(text)
	q = [item.lower() for item in q]
	txt = [item.lower() for item in txt]

	q = np.asarray(q)
	txt = np.asarray(txt)

	# Make lists of locations of each word in query
	lsts = [[] for i in range(len(q))]
	print(len(lsts))

	for i in range(len(q)):
		lsts[i] = list(np.where(txt == q[i])[0])
	print(lsts)

	# Initialize score to worst value
	score = -1


	# Calculate score of eligible paths
	for indices in itertools.product(*lsts):
		ind = list(indices)
		print(ind)

		if(sorted(ind) == ind):
			if(score > ind[len(ind)-1]-ind[0] or score == -1):
				score = ind[len(ind)-1]-ind[0]
		print(score)

	return 1/score

# Example:
# distance('World Cup', 'I have a cup in the world. Soccer is the most popular in the world. Therefore, it makes sense that the World Cup is so popular.')
# returns 1





