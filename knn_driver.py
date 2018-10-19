import numpy as np
import unittest
import random 


from utils import parse_images
from utils import parse_labels

from neural_net import NeuralNet

import json


x_train_path = 'data/X_train'
x_test_path = 'data/X_test'
y_train_path = 'data/y_train'
y_test_path = 'data/y_test'

class TestClassifiers(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		print "Setting up test environment..."
		print "\tParsing images"
		cls.X_train = parse_images(x_train_path)
		cls.X_test = parse_images(x_test_path)
		print "\tParsing labels"
		cls.y_train = parse_labels(y_train_path)
		cls.y_test = parse_labels(y_test_path)
		print "Finished setting up testing environment\n"
    	random.seed(0)
    	np.random.seed(0)



	def test_knn_classification_k_3(self):

		print "Constructing "


		learning_rate = 0.5
		structure = {'num_inputs': 784, 'num_hidden': 30, 'num_outputs': 10}
		candidate = NeuralNet(structure, learning_rate)

		#iterations = 15000
		iterations = 2

		trainX = np.array(self.X_train)#[:20000,:])
		trainY = np.array(self.y_train)#[:20000])

		candidate.train(trainX, trainY, iterations)

		cand_error = candidate.test(self.X_train[:100,:], self.y_train[:100])
		print "Train fraction: ", cand_error
		cand_error = candidate.test(self.X_test[:100,:], self.y_test[:100])
		print "Test fraction: ", cand_error


	
def main():
	suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestClassifiers)
	unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == "__main__":
	main()
