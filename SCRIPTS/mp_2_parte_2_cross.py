# --------------> Autores <-----------------
#       Claudia Cortés          11711357
#       Ingrid Dominguez        11711355
#
#librerias
import sys
import numpy as np
#Para Manejo de archivos
import pandas as pd
import csv
#Random forest 
from sklearn.ensemble import RandomForestClassifier
#Para crear los sets del crossvalidation
from sklearn.model_selection import train_test_split
#Evaluar el rendimiento 
from sklearn.metrics import precision_recall_fscore_support as score


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

def creatSets(datos,tags):
    rep=separar(datos,tags)
    tempsX=rep[0]
    tempsY=rep[1]
    tempsPred=rep[2]
    TempsVal=rep[3]
    F1Temps=[]
    confs = pd.read_csv('./confs/configuraciones.csv',engine='python') #Cargar configuraciones
    fd = open('./GRAFICAS/Salida.csv','a') #Salida de configuraciones
    fd.write('Criterio,Arboles,Profundidad,Atributos\n')
    for ind in confs.index: 
        criterio=str(confs['criterio'][ind])
        arboles=int(confs['arboles'][ind])
        profundidad=int(confs['profundidad'][ind])
        atributos=str(confs['atributos'][ind])
       
        # Creacion del bosque
        bosque = RandomForestClassifier(criterion=criterio,
            n_estimators=arboles,
            max_depth=profundidad,
            max_features=atributos)
        F1Temps=[]
        print("Generando Random Forest para configuracion ",ind)
        for j in range(5):
            for i in range(5): 
                bosque.fit(tempsX[i],tempsY[i])
                #Predicciones y metricas
                prediccion = bosque.predict(tempsPred[i]) 
                recision,recall,fscore,support=score(TempsVal[j], prediccion,average='macro')
                F1Temps.append(fscore)
            linea=criterio+','+str(arboles)+','+str(profundidad)+','+atributos+','+str(F1Temps).strip('[]')+'\n'
            F1Temps=[]
            fd.write("\n")
            fd.write(linea)
        fd.write("\n")
    fd.close() 
    return datos

#deficion de main#
def main():
    path = sys.argv[1]
    datos = pd.read_csv(path, engine='python')
    datos=preprocesar(datos)
    # Valores a predecir 
    tags=getTags(datos)
    #Limpiando y eliminando columnas de resultados
    datos=limpiarFeatures(datos)
    #one hot encoding de los demas attrs
    datos=oneHot(datos)
    creatSets(datos,tags)

#solo inicia si es el proceso inicial#
if __name__ == "__main__":
    main()
