""" Poker Round2: Este proyecto ya lo habia desarrollado pero el problema es que los porcentajes eran demaciado bajos, rondaba entre 50 y
40 es casi como una moneda al aire, por lo que no nos vamos a rendir y vamos a realizar bien el proyecto, ahora vamos a explicar en que 
consiste, en base a una data de poker realizaremos predicciones de que juego tenemos dependiendo la mano que tengamos. """

#Comenzamos importamos los modulos necesarios para el proyecto
import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split
#Siempre uso bosques aleatorios para realizar predicciones, ahora vamos a utilizar KNN
from sklearn.ensemble import RandomForestClassifier

#Vamos a sacar algunas estadisticas, esto es opcional
import matplotlib.pyplot as plt

#Importamos la data
DatosPoker = pd.read_csv('Data/poker-hand-training-true.csv')

#Creamos columnas para la data
DatosPoker.columns = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'R']

#Vemos la informacion de nuestra data
#Tipos de datos
print(DatosPoker.info())
#Suma de datos nulos
print(DatosPoker.isnull().sum())

#Separamos los datos en datos de entrenamiento y prueba
X = np.array(DatosPoker.drop(['R'], 1))
Y = np.array(DatosPoker['R'])

X_Entre, X_Prueba, Y_Entre, Y_Prueba = train_test_split(X, Y, test_size = 0.05)

#Implementamos nuestro algoritmo
Algoritmo = RandomForestClassifier()

#Entreanmos nuestro algoritmo
Algoritmo.fit(X_Entre, Y_Entre)

#Calculamos el score
print(Algoritmo.score(X_Prueba, Y_Prueba))

#Solo llegamos a las 62, me vencio :(