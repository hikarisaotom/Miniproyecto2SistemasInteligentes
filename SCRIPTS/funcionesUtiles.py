#Manejo de arreglos
import numpy as np
#Manejo de archivos
import pandas as pd
#Para crear los sets del crossvalidation
from sklearn.model_selection import train_test_split

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
    #Obtenemos los valores a predecir
    tags = np.array (datos ['clase'])
    return tags

def limpiarFeatures(datos):
    # Elimina las etiquetas de las columnas
    datos = datos.drop ('clase', axis = 1)
    return datos

def separar(datos,tags):
     #Sets para CrossValidation
    entrenamiento1, prueba1, entrenamiento2, prueba2 = train_test_split(datos, tags, test_size=0.50, random_state=66,shuffle=True) #dejamos el 50%  para prueba
    #segunda ronda de sets
    entrenamiento3, prueba3, entrenamiento4, prueba4 = train_test_split(datos, tags, test_size=0.50, random_state=70,shuffle=True) #dejamos el 50%  para prueba
    #Tercera ronda.
    entrenamiento5, prueba5,entrenamiento6, prueba6 = train_test_split(datos, tags, test_size=0.50, random_state=1,shuffle=True) #dejamos el 50%  para prueba
    #Cuarta ronda.
    entrenamiento7, prueba7,entrenamiento8, prueba8 = train_test_split(datos, tags, test_size=0.50, random_state=1,shuffle=True) #dejamos el 50%  para prueba
    #Quinta ronda.
    entrenamiento9, prueba9,entrenamiento10, prueba10 = train_test_split(datos, tags, test_size=0.50, random_state=1,shuffle=True) #dejamos el 50%  para prueba
    tempsX=[entrenamiento1,entrenamiento3,entrenamiento5,entrenamiento7,entrenamiento9]
    tempsY=[entrenamiento2,entrenamiento4,entrenamiento6,entrenamiento8,entrenamiento10]
    tempsPred=[prueba1,prueba3,prueba5,prueba7,prueba9]
    TempsVal=[prueba2,prueba4,prueba6,prueba8,prueba10]
    Respuesta=[tempsX,tempsY,tempsPred,TempsVal]
    return Respuesta

def separarCompleto(datos,tags):
     #Sets para CrossValidation
    entrenamiento1, prueba1, entrenamiento2, prueba2 = train_test_split(datos, tags, test_size=0.0000001, random_state=66,shuffle=True) #dejamos el 50%  para prueba
    #segunda ronda de sets
    print(entrenamiento1.shape)
    print(prueba1.shape)
    print(entrenamiento2.shape)
    print(prueba2.shape)
    return  entrenamiento1, prueba1, entrenamiento2, prueba2



def procesarDatos(datos,op):
    datos=preprocesar(datos)
    # Valores a predecir 
    tags=getTags(datos)
    #Limpiando y eliminando columnas de resultados
    datos=limpiarFeatures(datos)
    #one hot encoding de los demas attrs
    datos=oneHot(datos)
    if op==1:
        return separar(datos,tags)
    else: 
        return separarCompleto(datos,tags)
    

def cargarDatos(path):
    return pd.read_csv(path,engine='python')
