# --------------> Autores <-----------------
#       Claudia Cortés          11711357
#       Ingrid Dominguez        11711355
#
#Otras funciones
import sys
import numpy as np
#Para Manejo de archivos
import pandas as pd
#Para procesamiento de datos y demas.
import funcionesUtiles as funciones

#Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
#metricas
from sklearn.metrics import precision_recall_fscore_support as score

def Categorical(Crosssets,tags):
    tempsX=Crosssets[0]
    tempsY=Crosssets[1]
    tempsPred=Crosssets[2]
    TempsVal=Crosssets[3]
    #Categorical
    print("--------------------------->Categorical<---------------------------")
    naiveCate= CategoricalNB()
    naiveCate.fit(tempsX[0].abs(), tempsY[0])
    prediccion = naiveCate.predict(tempsPred[0].abs()) 
    fscore=score(TempsVal[0], prediccion,average='macro')
    funciones.estats(TempsVal[0], prediccion)
    print(" Categorical: F-1 ",fscore[0])

def Gausiano(Crosssets,tags):
    tempsX=Crosssets[0]
    tempsY=Crosssets[1]
    tempsPred=Crosssets[2]
    TempsVal=Crosssets[3]
    #Gausiano
    print("--------------------------->GAUSIANO<---------------------------")
    naiveGausiano = GaussianNB()
    naiveGausiano.fit(tempsX[0], tempsY[0])
    prediccion = naiveGausiano.predict(tempsPred[0]) 
    fscore=score(TempsVal[0], prediccion,average='macro')
    funciones.estats(TempsVal[0], prediccion)
    print(" Gausiano: F-1 ",fscore[0])

def Bernoulli(Crosssets,tags):
    tempsX=Crosssets[0]
    tempsY=Crosssets[1]
    tempsPred=Crosssets[2]
    TempsVal=Crosssets[3]
    #Bernoulli 
    print("--------------------------->Bernoulli<---------------------------")
    naiveBerno = BernoulliNB()
    naiveBerno.fit(tempsX[0], tempsY[0])
    prediccion = naiveBerno.predict(tempsPred[0]) 
    fscore=score(TempsVal[0], prediccion,average='macro')
    funciones.estats(TempsVal[0], prediccion)
    print(" Bernoulli: F-1 ",fscore[0])

#deficion de main#
def main():
    #path = sys.argv[1]
    path = './DATA/completo_train_synth_dengue.csv'
    datos = funciones.cargarDatos(path)
    procesado=funciones.procesarDatosNormalizados(datos,1)
    tags=funciones.getTags(datos,1)
    Categorical(procesado,tags)
    Gausiano(procesado,tags)
    Bernoulli(procesado,tags)


#solo inicia si es el proceso inicial#
if __name__ == "__main__":
    main()