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
#Evaluar el rendimiento 
from sklearn.metrics import classification_report, confusion_matrix
def predecir(bosque,datos):
    prediccion = bosque.predict(datos[0]) 
    print("------>ESTADISTICAS PARAEL RANDOM FOREST<------")
    print(classification_report(datos[1], prediccion))
    

#deficion de main
def main():
    #path = sys.argv[1]
    #path = sys.argv[2]
    path = './DATA/completo_train_synth_dengue.csv'
    nombre="./Archivos_salida/salida"
    datos = funciones.cargarDatos(path)
    procesado=funciones.procesarDatos(datos,2)
    bosque=funciones.cargarBinario(nombre+".rfc")
    predecir(bosque,procesado)
   

#solo inicia si es el proceso inicial#
if __name__ == "__main__":
    main()
 