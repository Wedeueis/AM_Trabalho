import csv
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV
import sys

###Pré processamento dos dados

#Lendo a base de dados de um arquivo csv
if len(sys.argv) != 2:
    print("Please inform the filename")
    exit(1)

fname = sys.argv[1]

try:
    #Lendo a base de dados de um arquivo csv
	csv_file_object = csv.reader(open(fname),
							delimiter=',', quotechar='"')
except IOError:
    print("File %s doesn't exist" % fname)
    exit(1)
data = []
for row in csv_file_object:
	data.append(row)

data = np.array(data)

#Separando os atributos das etiquetas
s = len(data[0])-1
X = data[:,:s]
X = X.astype(np.float)
y = data[:,s]
y = y.astype(np.float)

#Normalizando a amplitude
norm_x = X / X.max(axis=0)

###Treinamento e teste do modelo

#Laço para encontrar os k melhores atributos
kf = KFold(n_splits=10,shuffle=True)
print(kf)
print()
scrs = []
C = 1.0
errs = []
for k in range(1,s+1):
	#10-fold Cross Validation
	new_x = SelectKBest(chi2, k=k).fit_transform(norm_x, y)

	for train_index, test_index in kf.split(new_x):
		X_train, X_test = new_x[train_index], new_x[test_index]
		y_train, y_test = y[train_index], y[test_index]

		#Criando o modelo de regressão linear
		clf = svm.SVC(kernel='linear', C=C)
		#Treinando o modelo na base de treinamento
		clf.fit(X_train, y_train)
		#Desempenho na base de teste
		score = clf.score(X_test, y_test)
		#print("Score: %.2f" % score)

		scrs.append(score)

	#Média dos erros das 10 iterações
	sumx = 0

	for x in scrs:
		sumx += x

	mean_scr = sumx/len(scrs)

	errs.append(mean_scr)

	print("Score médio para o(s)", k, "melhor(es) atributo(s): %.2f" % mean_scr)
	print()

print("Erros para o i+1 atributos principais: \n", errs)

#Seleção de modelo: Busca dos melhores hiperparametros
max_ind = np.argmax(errs)

parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10, 100]}

new_x = SelectKBest(chi2, k=max_ind).fit_transform(norm_x, y)

svr = svm.SVC()
grid = GridSearchCV(svr, parameters, cv=10)
grid.fit(norm_x, y)
print()
print("Modelo final:")
print("Melhor score alcançado: %.2f" % grid.best_score_)
print("Parametros encontrados: ", grid.best_params_)
print()
