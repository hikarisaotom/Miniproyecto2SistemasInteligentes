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


def predecir(bosque,datos,tags):
    prediccion = bosque.predict(datos[0]) 
    funciones.estats(tags,prediccion)

#deficion de main
def main():
    path = sys.argv[1]
    nombre = sys.argv[2]
    datos = funciones.cargarDatos(path)
    tags=funciones.getTags(datos,1)
    procesado=funciones.procesarDatos(datos,2)
    bosque=funciones.cargarBinario(nombre+".rfc")
    predecir(bosque,procesado,tags)
    
   

#solo inicia si es el proceso inicial#
if __name__ == "__main__":
    main()
 