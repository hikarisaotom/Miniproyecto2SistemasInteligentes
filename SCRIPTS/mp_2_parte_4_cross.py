# --------------> Autores <-----------------
#       Claudia Cortés          11711357
#       Ingrid Dominguez        11711355
#
#Otras funciones
import sys
import numpy as np
#Para Manejo de archivos
import pandas as pd
# importing necessary libraries 
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 
#Naive Bayes
from sklearn.preprocessing import OrdinalEncoder

def Gausiano(datos):
    enc = OrdinalEncoder()
    X = [['Male', 1], ['Female', 3], ['Female', 2]]
    enc.fit(X)
    return 0

def Bernoulli():
    return 0

def CategoricalNB():
    return 0

def preprocesar(datos):
    datos=datos.replace(np.nan,2,regex=True)
    datos=datos.replace("NO","No",regex=True)
    return datos

def oneHot(datos):
    #One hot encoding
    datos = pd.get_dummies (datos)
    # Convertir a características de matriz 
    numpy = np.array (datos)
    return datos

def getTags(datos):
    #Convertimos todo en un arreglo
    tags1 = np.array (datos ['clase'])
    return oneHot(tags1)

def limpiarFeatures(datos):
    # Elimina las etiquetas de las columnas
    datos = datos.drop ('clase', axis = 1)
    return datos   


#deficion de main#
def main():
   #path = sys.argv[1]
    #path = './DATA/clinica_train_synth_dengue.csv'
    #path = './DATA/laboratorio_train_synth_dengue.csv'
    path = './DATA/completo_train_synth_dengue.csv'
    datos = pd.read_csv(path, engine='python')
    datos=preprocesar(datos)
    # Valores a predecir 
    tags=getTags(datos)
     #Limpiando y eliminando columnas de resultados
    datos=limpiarFeatures(datos)
    #one hot encoding de los demas attrs
    datos=oneHot(datos)
    # Lista de Features 
    features = list (datos.columns)
    print(datos.info())

#Gausiano(GaussianNB), Bernoulli (BernoulliNB), y Categórico (CategoricalNB)

#solo inicia si es el proceso inicial#
if __name__ == "__main__":
    main()