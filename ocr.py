import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from random import seed

from mlp import Layer, MLP

# read the data set
data = pd.read_csv("ocr_dataset.csv")

# separate atributtes from classes
values = data.iloc[0:210, 1:65].values
labels = data.iloc[0:210, 0].values

# Split the data set in train data set an test data set
train_values, test_values, train_labels, test_labels = train_test_split(values, labels, test_size=0.2, random_state=101)

train_values = train_values.tolist()
train_labels = train_labels.tolist()
test_values = test_values.tolist()
test_labels = test_labels.tolist()

train_labels_array = []
for label in train_labels:
	if label == 0:
		train_labels_array.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
	elif label == 1:
		train_labels_array.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
	elif label == 2:
		train_labels_array.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
	elif label == 3:
		train_labels_array.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
	elif label == 4:
		train_labels_array.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
	elif label == 5:
		train_labels_array.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
	elif label == 6:
		train_labels_array.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
	elif label == 7:
		train_labels_array.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
	elif label == 8:
		train_labels_array.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
	elif label == 9:
		train_labels_array.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

# Define o numero de entradas do primeiro layer
n_inputs = 64
# Define o numero de neuronios do layer de saída
n_outputs = 10
# Cada elemento do array define o numero de neuronios em um layer oculto
hidden_layers = []

teste = 1
seed(1)

for j in range(teste):

	hidden_layers = [20]

	# Define a rede a ser treinada
	train = MLP(n_inputs, n_outputs, hidden_layers)

	train.train(train_values, train_labels_array, 0.5, 0.9, 1000)

	for i in range(len(test_values)):
		row = test_values[i]
		prediction = train.classify(row)
		print('Expected=%d, Got=%d' % (test_labels[i], prediction))

	print ("###########################################")
