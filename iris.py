import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from mlp import Layer, MLP

# read the data set
data = pd.read_csv("iris_dataset.csv")

# Get data only from setosa and versicolor classes
# Get the atributtes sepal lenth and petal length
values = data.iloc[0:150, [0, 1, 2, 3]].values
labels = data.iloc[0:150, 4].values

# Redefine setosa class as 0 and versicolor as 1
labels = np.where(labels == "Iris-setosa", 0, (np.where(labels == "Iris-versicolor", 1, 2)))

# Split the data set in train data set an test data set
train_values, test_values, train_labels, test_labels = train_test_split(values, labels, test_size=0.2, random_state=101)

train_values = train_values.tolist()
train_labels = train_labels.tolist()
test_values = test_values.tolist()
test_labels = test_labels.tolist()

train_labels_array = []
for label in train_labels:
	if label == 0:
		train_labels_array.append([1, 0, 0])
	elif label == 1:
		train_labels_array.append([0, 1, 0])
	elif label == 2:
		train_labels_array.append([0, 0, 1])


# Define o numero de entradas do primeiro layer
n_inputs = 4
# Define o numero de neuronios do layer de saída
n_outputs = 3
# Cada elemento do array define o numero de neuronios em um layer oculto
hidden_layers = [10]

# Define a rede a ser treinada
train = MLP(n_inputs, n_outputs, hidden_layers)

train.train(train_values, train_labels_array, 0.5, 0.1, 1000)

for i in range(len(test_values)):
	row = test_values[i]
	prediction = train.classify(row)
	print('Expected=%d, Got=%d' % (test_labels[i], prediction))
