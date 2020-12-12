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

#metricas
from sklearn.metrics import precision_recall_fscore_support as score

def entrenar(Crosssets):
    tempsX=Crosssets[0]
    tempsY=Crosssets[1]
    tempsPred=Crosssets[2]
    TempsVal=Crosssets[3]
    F1Temps=[]
    naive = GaussianNB()
    naive.fit(tempsX[0], tempsY[0])
    prediccion = naive.predict(tempsPred[0]) 
    fscore=score(TempsVal[0], prediccion,average='macro')
    F1Temps.append(fscore[0])
    print(F1Temps)
   

#deficion de main#
def main():
    #path = sys.argv[1]
    path = './DATA/completo_train_synth_dengue.csv'
    datos = funciones.cargarDatos(path)
    procesado=funciones.procesarDatosNormalizados(datos,1)
    entrenar(procesado)
    print("SALE")

#Gausiano(GaussianNB), Bernoulli (BernoulliNB), y Categórico (CategoricalNB)

#solo inicia si es el proceso inicial#
if __name__ == "__main__":
    main()