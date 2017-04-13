import csv
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import normalize

csv_file_object = csv.reader(open('winequality-red.csv'), 
						delimiter=';', quotechar='"')
data = []
for row in csv_file_object:
	data.append(row)

data = np.array(data)

X = data[1:,:11]
X = X.astype(np.float)
Y = data[1:,11]
Y = Y.astype(np.float)

norm_x = []
for i in X:
	norm_x.append(normalize(i[:,np.newaxis], axis=0).ravel())

clf = MLPRegressor(hidden_layer_sizes=(11,), early_stopping=True)

clf.fit(norm_x, Y)