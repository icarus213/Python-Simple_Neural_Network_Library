from numpy import exp, array, random, dot, average, save, load
import sys

if sys.version_info >= (3, 0):
	xrange = range

class NeuronLayer():
	def __init__(self, neurons, inputs, predefined_weight=array(0)):
		if predefined_weight.all() == 0:
			self.synaptic_weight = 2*random.random([inputs, neurons])
		else: self.synaptic_weight = predefined_weight

class NeuralNetwork():
	def __init__(self, layers):
		previousValue = 0
		for x in layers:
			if previousValue != 0:
				if len(previousValue.synaptic_weight[0]) != len(x.synaptic_weight):
					raise Exception('The number of inputs on a layer must be equal to the number of neurons on the previous layer')
			previousValue = x
		self.layer = layers

	def __sigmoid(self, x, deriv=False):
		if(deriv):
			return x * (1 - x)
		else:
			return 1 / (1 + exp(-x))

	def think(self, data_input):
		previousValue = data_input
		output_from_layer = []
		for x in self.layer:
			output_from_layer.append(self.__sigmoid(dot(previousValue,x.synaptic_weight)))
			previousValue = output_from_layer[-1]
		return output_from_layer

	def train(self, training_set_input, training_set_output, number_of_training_iterations, tolerance=0, learning_rate=1, verbose=0):
		if len(training_set_input) == len(training_set_output):
			for iteration in xrange(number_of_training_iterations):
				output_from_layer = self.think(training_set_input)
				layer_error = [training_set_output - output_from_layer[-1]]
				layer_delta = [layer_error[-1] * self.__sigmoid(output_from_layer[-1], deriv=True)]
				for x, y in reversed(list(enumerate(self.layer))):
					if x == 0: break
					layer_error.append(layer_delta[-1].dot(y.synaptic_weight.T))
					layer_delta.append(layer_error[-1] * self.__sigmoid(output_from_layer[x-1], deriv=True))
				layer_delta.reverse()
				layer_adjustment = []
				previousValue = training_set_input
				for x in xrange(len(self.layer)):
					layer_adjustment.append(previousValue.T.dot(layer_delta[x]))
					previousValue = output_from_layer[x]
				for x in xrange(len(self.layer)):
					self.layer[x].synaptic_weight += learning_rate * layer_adjustment[x]
				sum_error = average(sum([(training_set_output[i]-output_from_layer[-1][i])**2 for i in xrange(len(training_set_output))])/len(training_set_output))
				if sum_error < tolerance:
					break
				if verbose:
					print('>epoch=%d, learning rate=%.3f, error=%.3f' % (iteration, learning_rate, 100*sum_error))
		else:
			raise Exception('The number of training_set_input must be equal with the number of training_set_output')

	def print_weights(self):
		y = 0
		for x in self.layer:
			print('Layer ' + str(y))
			print(x.synaptic_weight)
			y += 1

	def get_weights(self):
		output = []
		for x in self.layer:
			output.append(x.synaptic_weight)
		return array(output)

	def load_weights(self,file_name):
		file = open(file_name, "r")
		c = load(file)
		if len(c) == len(self.layer):
			for x in xrange(len(c)):
				if len(c[x]) == len(self.layer[x].synaptic_weight):
					self.layer[x].synaptic_weight = array(c[x])
				else:
					raise Exception('The number of weights must match with the number of configured weights on layer ' + str(x) + '\n'
					 + 'The number of weights in file: ' + str(len(c[x])) + '. The number of configured weights: ' + str(len(self.layer[x].synaptic_weight)))
		else:
			raise Exception('The number of layers must match with the number of configured layers in the NeuralNetwork' + '\n'
			 + 'The number of layers in file: ' + str(c) + 'The number of layers in the NeuralNetwork: ' + str(len(self.layer)))
		file.close()

	def save_weights(self, file_name):
		file = open(file_name, "w")
		save(file,self.get_weights())
		file.close()
