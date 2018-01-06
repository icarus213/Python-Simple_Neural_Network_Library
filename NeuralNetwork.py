from numpy import exp, array, random, dot, average, savez, load
import sys

if sys.version_info >= (3, 0):
	xrange = range	# In python3 range is python2's xrange

class NeuronLayer():
	def __init__(self, neurons=0, inputs=0, predefined_weight=array(0)):
		if predefined_weight.all() == 0: 
			assert neurons != 0
			assert inputs != 0
			self.synaptic_weight = 2*random.random([inputs, neurons]) # Generate random weights
			self.neurons = neurons
			self.inputs = inputs
		else:
			self.synaptic_weight = predefined_weight 	# Load from a trained weights (load_weights is much preferred. Use this if you want to load weights for one layer only)
														# The value of 'neurons' and 'inputs' variables are ignored if you load weights this way
			self.neurons = predefined_weight.shape[1]
			self.inputs = predefined_weight.shape[0]


class NeuralNetwork():
	def __init__(self, layers):
		previousValue = 0
		for x in layers:
			if previousValue != 0:
				if len(previousValue.synaptic_weight[0]) != len(x.synaptic_weight):
					raise Exception('The number of inputs on a layer must be equal to the number of neurons on the previous layer')
			previousValue = x
		self.layer = layers
		self.total_layers = len(layers)
		self.neurons = [x.neurons for x in layers]
		self.inputs = [x.inputs for x in layers]

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
				sum_error = average(sum(array(layer_error[0])**2)/len(training_set_output))
				if sum_error < tolerance:
					break
				if verbose:
					print('>epoch=%d, learning rate=%.3f, error=%.3f' % (iteration, learning_rate, 100*sum_error) + '%')
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
		return output

	def load_weights(self,file_name):
		if '.npz' != file_name[-4:]:
			file_name += '.npz'
		if sys.version_info >= (3, 0):
			archive = load(file_name,encoding='latin1')
		else:
			archive = load(file_name) # Load the archive
		c = []
		for x in reversed(archive.files):
			c.append(archive[x]) # Load the weights from the files in the archive
		if len(c) == self.total_layers:
			for x in xrange(len(c)):
				if c[x].shape == self.layer[x].synaptic_weight.shape:
					self.layer[x].synaptic_weight = array(c[x])
				else:
					raise Exception('Different neural network configuration received from file:' + '\n'
						+ 'NeuralNetwork configuration: Layer %d: Neuron %d Input %d' % (x,self.neurons[x],self.inputs[x]) + '\n'
						+ 'NeuralNetwork configuration from file: Layer %d: Neuron %d Input %d' % (x,c[x].shape[1],c[x].shape[0]))
			raise Exception('The number of layers must match with the number of configured layers in the NeuralNetwork' + '\n'
			 + 'The number of layers in file: %d. The number of layers in the NeuralNetwork: %d' % (len(c),self.total_layers))

	def save_weights(self, file_name):
		savez(file_name,*self.get_weights()) # Save the weights to an archive

	@staticmethod
	def from_file(file_name):
		if '.npz' != file_name[-4:]:
			file_name += '.npz'
		if sys.version_info >= (3, 0):
			archive = load(file_name,encoding='latin1')
		else:
			archive = load(file_name) # Load the archive
		layers = []
		for x in reversed(archive.files):
			layers.append(NeuronLayer(archive[x])) # Load the weights from the files in the archive
		return NeuralNetwork(layers)
