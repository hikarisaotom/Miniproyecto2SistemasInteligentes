# --------------> Autores <-----------------
#       Claudia Cortés          11711357
#       Ingrid Dominguez        11711355
#

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
# training a linear SVM classifier 
from sklearn.svm import SVC 


def preprocesar(datos):
    datos=datos.replace(np.nan,2,regex=True)
    datos=datos.replace("NO","No",regex=True)
    return datos

 


#deficion de main#
def main():
   #path = sys.argv[1]
    #path = './DATA/clinica_train_synth_dengue.csv'
    #path = './DATA/laboratorio_train_synth_dengue.csv'
    path = './DATA/completo_train_synth_dengue.csv'
    datos = pd.read_csv(path, engine='python')
 

#solo inicia si es el proceso inicial#
if __name__ == "__main__":
    main()


