# --------------> Autores <-----------------
#       Claudia Cortés          11711357
#       Ingrid Dominguez        11711355
#

#librerias#
import sys
from time import time
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, mean_squared_error, r2_score


#deficion de main#
def main():
   #archivo1 = sys.argv[1]
    archivo1 = 'clinica_train_synth_dengue.csv'
    archivo2 = 'laboratorio_train_synth_dengue.csv'
    archivo3 = 'completo_train_synth_dengue.csv'
    datos1 = pd.read_csv(archivo1, engine='python')
    datos2 = pd.read_csv(archivo2, engine='python')
    datos3 = pd.read_csv(archivo3, engine='python')
    #para ver informacion
    print("DATOS DE CLINICA")
    datos1.info()
    print("-------------------------------------------------------------------")
    print("DATOS DE LAB")
    datos2.info()
    print("-------------------------------------------------------------------")
    print("DATOS DE CLINICA Y LAB")
    datos3.info()

#solo inicia si es el proceso inicial#
if __name__ == "__main__":
    main()

    #random forest
    #MSVMaquinas Soporte Vecto
    #Bayesiano ingenuo

    #Clases:
    # No dengue
    # Dengue no grave y sin isgos de alarma
    #Dengue no grave con signos de alarma
    #Dengue Grave