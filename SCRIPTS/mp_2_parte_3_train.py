# --------------> Autores <-----------------
#       Claudia Cortés          11711357
#       Ingrid Dominguez        11711355
#
import sys
#Para procesamiento de datos y demas.
import funcionesUtiles as funciones
#Para SVC
from sklearn.svm import SVC as SVC
#metricas
from sklearn.metrics import precision_recall_fscore_support as score

def entrenar(Datos):
    X=Datos[0]
    Y=Datos[1]
    model = SVC( C = 1,kernel = 'rbf',gamma = 'scale')
    return model.fit(X,Y)

#deficion de main
def main():
    path = sys.argv[1]
    nombre = sys.argv[2]
    datos = funciones.cargarDatos(path)
    procesado=funciones.procesarDatosNormalizados(datos,2)
    modeloSVM=entrenar(procesado)
    print("SVM Entrenado Exitosamente")
    funciones.GuardarBinario(modeloSVM,nombre+".svm")

#solo inicia si es el proceso inicial#
if __name__ == "__main__":
    main()
