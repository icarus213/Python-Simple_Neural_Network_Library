from NeuralNetwork import *
from numpy import array

# Create layer 0 (4 neurons, each with 3 inputs)
layer0 = NeuronLayer(4, 3)
# Create layer 1 (a single neuron with 4 inputs)
layer1 = NeuronLayer(1, 4)

# Combine the layers to create a neural network
neural_network = NeuralNetwork([layer0,layer1])

print "Stage 1) Random starting synaptic weights: "
neural_network.print_weights()

# The training set. We have 7 examples, each consisting of 3 input values
# and 1 output value.
training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1],[0,0,0],[1,0,0],[0,1,0]])
training_set_outputs = array([[0],[1],[1],[0],[0],[0],[0]])

# Train the neural network using the training set.
# Do it 60,000 times and make small adjustments each time.
neural_network.train(training_set_inputs, training_set_outputs, 60000, verbose=True)

print "Stage 2) New synaptic weights after training: "
neural_network.print_weights()

# Test the neural network with a new situation.
print "Stage 3) Considering a new situation [1, 1, 0] -> ?: "
output = neural_network.think(array([1,1,0]))
print output[-1] # The last item on the list is what you want