# --------------> Autores <-----------------
#       Claudia Cortés          11711357
#       Ingrid Dominguez        11711355
#
#librerias
#Otras funciones
import sys
import numpy as np
#Para Manejo de archivos
import pandas as pd
#Random forest 
from sklearn.ensemble import RandomForestClassifier
#Para crear los sets del crossvalidation
from sklearn.model_selection import train_test_split
#Evaluar el rendimiento 
from sklearn.metrics import classification_report, confusion_matrix


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
   # tags= oneHot(tags)
    return tags

def limpiarFeatures(datos):
    # Elimina las etiquetas de las columnas
    datos = datos.drop ('clase', axis = 1)
    return datos   

def creatSets(datos,tags,features):
    ##VALORES A CAMBIAR 
    #criterio='gini'
    criterio='entropy'
    arboles=200
    profundidad=15
    atributos=17

   # Sets para CrossValidation
    entrenamiento1, prueba1, entrenamiento2, prueba2 = train_test_split(
        datos,tags, test_size=0.40, random_state=66)
    
    # Creacion del bosque
    bosque = RandomForestClassifier(criterion=criterio,
                n_estimators=arboles,
                max_depth=profundidad,
                max_features=atributos)
    bosque.fit(entrenamiento1,entrenamiento2)
    #Predicciones y metricas
    prediccion = bosque.predict(prueba1) 
    print("------>ESTADISTICAS PARAEL RANDOM FOREST<------")
    print(classification_report(prueba2, prediccion))
    print('\n')
    
    return datos

#deficion de main#
def main():
   #path = sys.argv[1]
    path = './DATA/clinica_train_synth_dengue.csv'
    #path = './DATA/laboratorio_train_synth_dengue.csv'
    #path = './DATA/completo_train_synth_dengue.csv'
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
    #print(datos.info())
    creatSets(datos,tags,features)


#solo inicia si es el proceso inicial#
if __name__ == "__main__":
    main()
