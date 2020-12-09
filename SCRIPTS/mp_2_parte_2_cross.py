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
from sklearn.metrics import f1_score
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
   # tags= oneHot(tags)
    return tags

def limpiarFeatures(datos):
    # Elimina las etiquetas de las columnas
    datos = datos.drop ('clase', axis = 1)
    return datos   

def creatSets(datos,tags,features):
    ##VALORES A CAMBIAR 
    #criterio='gini'
   
     #Sets para CrossValidation
    entrenamiento1, prueba1, entrenamiento2, prueba2 = train_test_split(
    datos, tags, test_size=0.50, random_state=66,shuffle=True) #dejamos el 50%  para prueba
    #segunda ronda de sets
    entrenamiento3, prueba3, entrenamiento4, prueba4 = train_test_split(
    datos, tags, test_size=0.50, random_state=70,shuffle=True) #dejamos el 50%  para prueba
    #Tercera ronda.
    entrenamiento5, prueba5,entrenamiento6, prueba6 = train_test_split(
    datos, tags, test_size=0.50, random_state=1,shuffle=True) #dejamos el 50%  para prueba
    #Cuarta ronda.
    entrenamiento7, prueba7,entrenamiento8, prueba8 = train_test_split(
    datos, tags, test_size=0.50, random_state=1,shuffle=True) #dejamos el 50%  para prueba
    #Quinta ronda.
    entrenamiento9, prueba9,entrenamiento10, prueba10 = train_test_split(
    datos, tags, test_size=0.50, random_state=1,shuffle=True) #dejamos el 50%  para prueba

    tempsX=[entrenamiento1,entrenamiento3,entrenamiento5,entrenamiento7,entrenamiento9]
    tempsY=[entrenamiento2,entrenamiento4,entrenamiento6,entrenamiento8,entrenamiento10]
    tempsPred=[prueba1,prueba3,prueba5,prueba7,prueba9]
    TempsVal=[prueba2,prueba4,prueba6,prueba8,prueba10]
    F1Temps=[]
    confs=[]
    cont=0
    for conf in range(15):
        criterio='entropy'
        arboles=200
        profundidad=15
        atributos=17
        # Creacion del bosque
        bosque = RandomForestClassifier(criterion=criterio,
            n_estimators=arboles,
            max_depth=profundidad,
            max_features=atributos)
        for i in range(5): 
                bosque.fit(tempsX[i],tempsY[i])
                #Predicciones y metricas
                prediccion = bosque.predict(tempsPred[i]) 
                print("------>ESTADISTICAS PARAEL RANDOM FOREST<------")
                print(classification_report(TempsVal[i], prediccion))
                precision,recall,fscore,support=score(TempsVal[i], prediccion,average='macro')
                print('F-score   : {}'.format(fscore))
                print('\n')
                F1Temps.append(fscore)
    print("CONTADOR: ",cont)
 
              

    

    

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
