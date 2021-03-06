# --------------> Autores <-----------------
#       Claudia Cortés          11711357
#       Ingrid Dominguez        11711355
#
import sys
#Para procesamiento de datos y demas.
import funcionesUtiles as funciones
#Para SVC
from sklearn.svm import SVC as SVC

def predecir(bosque,datos,tags):
    prediccion = bosque.predict(datos[0]) 
    funciones.estats(tags,prediccion)

#deficion de main
def main():
    path = sys.argv[1]
    nombre = sys.argv[2]
    datos = funciones.cargarDatos(path)
    tags=funciones.getTags(datos,2)
    procesado=funciones.procesarDatosNormalizados(datos,2)
    modeloSVM=funciones.cargarBinario(nombre+".svm")
    predecir(modeloSVM,procesado,tags)

#solo inicia si es el proceso inicial#
if __name__ == "__main__":
    main()
