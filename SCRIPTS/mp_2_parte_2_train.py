# --------------> Autores <-----------------
#       Claudia Cortés          11711357
#       Ingrid Dominguez        11711355
#
#librerias
import sys
import numpy as np
#Random forest 
from sklearn.ensemble import RandomForestClassifier
#Para procesamiento de datos y demas.
import funcionesUtiles as funciones

#deficion de main
def main():
    #path = sys.argv[1]
    path = './DATA/completo_train_synth_dengue.csv'
    datos = funciones.cargarDatos(path)
    print("ENTRA")
    procesado=funciones.procesarDatos(datos,2)
    print("SAIO")
    

#solo inicia si es el proceso inicial#
if __name__ == "__main__":
    main()