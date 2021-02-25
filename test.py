import numpy as np
from math import exp
from mlp import Layer, MLP

inputs = [	[2.7810836,		2.550537003],
			[1.465489372,	2.362125076],
			[3.396561688,	4.400293529],
			[1.38807019,	1.850220317],
			[3.06407232,	3.005305973],
			[7.627531214,	2.759262235],
			[5.332441248,	2.088626775],
			[6.922596716,	1.77106367],
			[8.675418651,	-0.242068655],
			[7.673756466,	3.508563011]
		]

labels = [	[1, 0],
			[1, 0],
			[1, 0],
			[1, 0],
			[1, 0],
			[0, 1],
			[0, 1],
			[0, 1],
			[0, 1],
			[0, 1], ]

labels2 = [	0,
			0,
			0,
			0,
			0,
			1,
			1,
			1,
			1,
			1,]

# Define o numero de entradas do primeiro layer
n_inputs = 2
# Define o numero de neuronios do layer de saída
n_outputs = 2
# Cada elemento do array define o numero de neuronios em um layer oculto
hidden_layers = [4]

# Define a rede a ser treinada
train = MLP(n_inputs, n_outputs, hidden_layers)

train.train(inputs, labels, 0.5, 20)

for i in range(len(inputs)):
	row = inputs[i]
	prediction = train.classify(row)
	print('Expected=%d, Got=%d' % (labels2[i], prediction))
