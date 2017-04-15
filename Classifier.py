import csv
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold

###Pré processamento dos dados

#Lendo a base de dados de um arquivo csv
csv_file_object = csv.reader(open('winequality-white.csv'), 
						delimiter=',', quotechar='"')
data = []
for row in csv_file_object:
	data.append(row)

data = np.array(data)

#Separando os atributos das etiquetas
X = data[:,:11]
X = X.astype(np.float)
y = data[:,11]
y = y.astype(np.float)

#Normalizando a amplitude
norm_x = X / X.max(axis=0)

###Treinamento e teste do modelo

#10-fold Cross Validation
kf = KFold(n_splits=10)
kf.get_n_splits(norm_x)
print(kf)
scrs = []
C = 1.0

for train_index, test_index in kf.split(norm_x):
	X_train, X_test = norm_x[train_index], norm_x[test_index]
	y_train, y_test = y[train_index], y[test_index]

	#Criando o modelo de regressão linear
	clf = svm.SVC(kernel='linear', C=C)
	#Treinando o modelo na base de treinamento
	clf.fit(X_train, y_train)
	#Desempenho na base de teste
	score = clf.score(X_test, y_test)
	print("Score: %.2f" % score)

	scrs.append(score)

###Resultados

#Média dos erros das 10 iterações
sumx = 0

for x in scrs:
	sumx += x

mean_scr = sumx/len(scrs)

print("Score médio: %.2f" % mean_scr)