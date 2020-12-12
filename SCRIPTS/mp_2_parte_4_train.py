# --------------> Autores <-----------------
#       Claudia Cortés          11711357
#       Ingrid Dominguez        11711355
#
import sys
#Para procesamiento de datos y demas.
import funcionesUtiles as funciones
#Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
#metricas
from sklearn.metrics import precision_recall_fscore_support as score


def Categorical(Datos):
    naiveCate= CategoricalNB()
    return  naiveCate.fit(Datos[0].abs(), Datos[1])

def Gausiano(Datos):
    naiveGausiano = GaussianNB()
    return naiveGausiano.fit(Datos[0].abs(), Datos[1])


def Bernoulli(Datos):
    naiveBerno = BernoulliNB()
    return naiveBerno.fit(Datos[0].abs(), Datos[1])

#deficion de main
def main():
    path = sys.argv[1]
    naive=sys.argv[2]
    nombre = sys.argv[3]
    datos = funciones.cargarDatos(path)
    procesado=funciones.procesarDatosNormalizados(datos,2)
    opcion=[]
    bandera=True
    if naive==1:#Categorical
        opcion=Categorical(procesado)
    elif naive==2:#Bernoulli
        opcion=Bernoulli(procesado)
    elif naive==3:#Gausiano
        opcion=Gausiano(procesado)
    else:
        print("Opcion Invalida.")
        bandera=False
    if bandera:
            print("NAIVE Entrenado Exitosamente")
            funciones.GuardarBinario(opcion,nombre+".nb")

#solo inicia si es el proceso inicial#
if __name__ == "__main__":
    main()
