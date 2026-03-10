import numpy as np
def candidate_elimination(examples):
	num_attributes = len(examples[0]) - 1 # Number of attributes in the examples
	specific_hypothesis = ['0'] * num_attributes # most specific hypothesis
	general_hypothesis = ['?'] * num_attributes # most general hypothesis
	for example in examples:
		if example[-1] == 'Yes': # positive example
			for i in range(num_attributes):
				if specific_hypothesis[i] != example[i]:
					specific_hypothesis[i] = '?'
			# Remove inconsistent hypotheses from general_hypothesis
			for j in range(num_attributes):
				if example[j] != specific_hypothesis[j]:
					general_hypothesis[j] = specific_hypothesis[j]
		else: # negative example
			for i in range(num_attributes):
				if example[i] != specific_hypothesis[i]:
					general_hypothesis[i] = specific_hypothesis[i]
	# Remove inconsistent hypotheses from general_hypothesis
	temp = []
	for hypothesis in general_hypothesis:
		if hypothesis not in temp:
			temp.append(hypothesis)
	general_hypothesis = temp
	return specific_hypothesis, general_hypothesis
# Example dataset
dataset = [
['Some', 'Small', 'No', 'Affordable', 'Few', 'No', 'No'],
['Many', 'Big', 'No', 'Expensive', 'Many', 'Yes', 'Yes'],
['Many', 'Medium', 'No', 'Expensive', 'Few', 'Yes', 'Yes'],
['Many', 'Small', 'No', 'Affordable', 'Many', 'Yes', 'Yes']
]
# Applying the Candidate-Elimination algorithm
specific, general = candidate_elimination(dataset)
print("Specific hypothesis:", specific)
print("General hypotheses:", general)