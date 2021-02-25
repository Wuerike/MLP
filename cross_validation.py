import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from statistics import mean 
import pandas as pd
import numpy as np
import random

from mlp import Layer, MLP

print("Entradas: 64")
print("Neuronios no primeiro layer: 10")
print("Neuronios no ultimo layer: 10")
print("Taxa de aprendizagem: 0.5")
print("Taxa de momentum: 0.5")
print("Número de épocas: 500")
print()

# read the data set
data = pd.read_csv("ocr_dataset.csv")

# separate atributtes from classes
values = data.iloc[0:210, 1:65].values
labels = data.iloc[0:210, 0].values

skf = StratifiedKFold(n_splits=10, shuffle=True)
skf.get_n_splits(values, labels)

accuracy = []

for train_index, test_index in skf.split(values, labels):

	test_values = []
	test_labels = []
	for index in test_index:
		test_values.append(values[index].tolist())
		test_labels.append(labels[index].tolist())

	train_values = []
	train_labels = []
	for index in train_index:
		train_values.append(values[index].tolist())
		train_labels.append(labels[index].tolist())

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
	hidden_layers = [10]

	# Define a rede a ser treinada
	train = MLP(n_inputs, n_outputs, hidden_layers)

	train.train(train_values, train_labels_array, 0.5, 0.5, 500)

	match = 0
	for j in range(len(test_values)):
		row = test_values[j]
		prediction = train.classify(row)
		if test_labels[j] == prediction:
			match += 1
		print('Expected: %d, Got: %d' % (test_labels[j], prediction))

	accuracy.append(match/(j+1))
	print("Accuracy: ", match/(j+1))
	print()
	print("################################################")
	print()

print ("Mean accuracy: ", mean(accuracy))
