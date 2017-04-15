import csv
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import normalize

#Lendo a base de dados de um arquivo csv
csv_file_object = csv.reader(open('winequality-white.csv'), 
						delimiter=',', quotechar='"')
data = []
for row in csv_file_object:
	data.append(row)

data = np.array(data)

#Separando os atributos das etiquetas
X = data[1:,:11]
X = X.astype(np.float)
Y = data[1:,11]
Y = Y.astype(np.float)

#Normalizando a amplitude
norm_x = X / X.max(axis=0)

#Separando o conjunto de treinamento do de teste
X_train = X[:-1200]
X_test = X[-1200:]

print('Tamanho do conjunto de treinamento: \n', len(X_train))
print('Tamanho do conjunto de teste: \n', len(X_test))

Y_train = Y[:-1200]
Y_test = Y[-1200:]

#Criando o modelo de regressão linear
regr = linear_model.LinearRegression()

#Treinando o modelo na base de treinamento
regr.fit(X_train, Y_train)

#Coeficientes da reta aprendida
print('Coefficients: \n', regr.coef_)

#Erro médio quadrático
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_test) - Y_test) ** 2))

#Pontuação em variancia explicada: 1 é uma previsão perfeita
print('Variance score: %.2f' % regr.score(X_test, Y_test))