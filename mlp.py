import numpy as np
from random import random
from math import exp


class Layer(object):

	def __init__(self, n_inputs, n_neurons):
		# Number of inputs
		self.n_inputs = n_inputs
		# Number of neurons
		self.n_neurons = n_neurons
		# Array to store each neuron summation
		self.sum_array = np.array([])
		# Input data array
		self.inputs = []
		# Array with each neuron bias (initialized with random number)
		self.bias = np.array([0.1 for i in range(self.n_neurons)])
		# 2D array with the weights for each input for each neuron (initialized with random number)
		self.weights = np.array([[0.1 for i in range(self.n_inputs)] for i in range(self.n_neurons)])
		# Error signal for each neuron, used when backpropagating
		self.delta = np.zeros(n_neurons)
		# Weight variation from previous interation
		self.prev_delta_weight = np.zeros(n_neurons)
		# Array with the layer neurons output
		self.output_array = []

	# Calculate the logistic function for X
	def logistic(self, x):
		return 1.0/ (1.0 + exp(-x))

	# For each neuron makes: scalar product between weights and inputs + bias value
	def build_sum_array(self):
		self.sum_array = (self.weights.dot(self.inputs) + self.bias)

	# Set this layer inputs
	def set_inputs(self, inputs):
		self.inputs = inputs

	# Calculate this layer output
	def calculate_output(self):
		output = []
		self.build_sum_array()
		for x in self.sum_array:
			activativation = self.logistic(x)
			output.append(activativation)
		self.output_array = output


class MLP(object):

	def __init__(self, n_inputs, n_outputs, hidden_layers):
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs
		self.hidden_layers = hidden_layers
		self.network = []

		layers = hidden_layers
		layers.append(self.n_outputs)

		for i in range(len(layers)):
			if i == 0:
				self.network.append(Layer(n_inputs, layers[i]))
			else:
				self.network.append(Layer(layers[i-1], layers[i]))

	# Calculate the derivative of the logistic function for X
	def logistic_deriv(self, x):
		return x * (1.0 - x)

	# Forward propagate input to a network output
	def forward_propagate(self, input_row):
		# Run through the layers from 0 to n
		for i in range(len(self.network)):
			layer = self.network[i] 
			# Uses row as first layer input
			if i == 0:
				layer.set_inputs(input_row)
			else:
				behind_layer = self.network[i-1]
				layer.set_inputs(behind_layer.output_array)
			layer.calculate_output()

	# Back propagate error and store in neurons
	def backward_propagate_error(self, expected):
		# Run through the layers from the end to the beginning
		for i in reversed(range(len(self.network))):
			layer = self.network[i]
			output = layer.output_array
			errors = list()

			# If is the output layer
			if i == len(self.network)-1:
				for j in range(len(output)):
					errors.append((expected[j] - output[j]))
			# Or if is some hidden layer	
			else:
				ahead_layer = self.network[i+1]
				# Run through the neurons from this layer
				for j in range(len(output)):
					error = 0.0
					# Run through the neurons from ahead layer
					for k in range(ahead_layer.n_neurons):
						error += (ahead_layer.weights[k][j] * ahead_layer.delta[k])
					errors.append(error)
			for k in range(len(output)):
				layer.delta[k] = errors[k] * self.logistic_deriv(output[k])

	# Update network weights with error
	def update_weights(self, inputs, l_rate, momentum):
		# Run through the layers
		for i in range(len(self.network)):
			layer = self.network[i]
			if i != 0:
				inputs = [output for output in self.network[i - 1].output_array]
			for neuron in range(layer.n_neurons):
				for j in range(len(inputs)):
					layer.weights[neuron][j] += ((l_rate * layer.delta[neuron] * inputs[j]) + (momentum * layer.prev_delta_weight[neuron]))
					layer.prev_delta_weight[neuron] = (l_rate * layer.delta[neuron] * inputs[j])
				layer.bias[neuron] += l_rate * layer.delta[neuron]

	# Train a network for a fixed number of epochs
	def train(self, inputs, labels, l_rate, momentum, n_epoch):
		for epoch in range(n_epoch):
			sum_error = 0
			for i in range(len(inputs)):
				row = inputs[i]
				self.forward_propagate(row)
				outputs = self.network[-1].output_array
				sum_error += sum([(labels[i][j]-outputs[j])**2 for j in range(len(labels[i]))])
				self.backward_propagate_error(labels[i])
				self.update_weights(row, l_rate, momentum)
			print('epoch: %d, error: %.3f' % (epoch, sum_error))

		print("--------------")
		for i in range(len(self.network)):
			layer = self.network[i]
			print("Layer: ", i)
			print("Weights: ")
			print(layer.weights)
			print("--------------")

	# Make a prediction with a network
	def classify(self, row):
		self.forward_propagate(row)
		self.network[-1].calculate_output()
		outputs = self.network[-1].output_array
		return outputs.index(max(outputs))