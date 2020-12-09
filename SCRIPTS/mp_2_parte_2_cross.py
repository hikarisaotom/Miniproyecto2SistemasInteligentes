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
#from sklearn.ensemble import RandomForestRegressor
#from sklearn import model_selection

#Para crear los sets del crossvalidation
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
#Evaluar el rendimiento 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

#preprocesamiento
from sklearn import preprocessing
from sklearn import utils


def preprocesar(datos):
    datos=datos.replace(np.nan,2,regex=True)
    datos=datos.replace("NO","No",regex=True)
    """datos=datos.replace("No",0,regex=True)
    datos=datos.replace("Si",1,regex=True)
    datos=datos.replace("Positiva",1,regex=True)
    datos=datos.replace("Negativa",0,regex=True)
    datos=datos.replace("F",1,regex=True)
    datos=datos.replace("M",0,regex=True)
    datos=datos.replace("Persistente",1.5,regex=True)"""
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
    datos = datos.drop ('dias_fiebre', axis = 1)
    datos = datos.drop ('dias_ultima_fiebre', axis = 1)
    return datos   

def creatSets(datos,tags,features):
    
   # implementing train-test-split
    X_train, X_test, y_train, y_test = train_test_split(datos,tags, test_size=0.33, random_state=66)
    # random forest model creation
    rfc = RandomForestClassifier()
    rfc.fit(X_train,y_train)
    #predictions
    rfc_predict = rfc.predict(X_test)

    rfc_cv_score = cross_val_score(rfc, datos,tags, cv=10, scoring='roc_auc')

  
    print("=== Classification Report ===")
    print(classification_report(y_test, rfc_predict))
    print('\n')

      # Instantiate model with 1000 decision trees
   # rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    #rf.fit(X_train,y_train)
    #predictions = rf.predict(X_test)
    
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
    #print(datos.info())
    creatSets(datos,tags,features)
    print("--->SALE")


#solo inicia si es el proceso inicial#
if __name__ == "__main__":
    main()
