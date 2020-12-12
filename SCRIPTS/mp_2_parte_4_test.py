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


def Categorical(datos,modelo):
    #Categorical
    print("--------------------------->Categorical<---------------------------")
    prediccion = modelo.predict(datos[0].abs()) 
    fscore=score(datos[1], prediccion,average='macro')
    funciones.estats(datos[1], prediccion)
    print(" Categorical: F-1 ",fscore[0])
    return fscore[0]

def Gausiano(datos,modelo):
    #Gausiano
    print("--------------------------->Gausiano<---------------------------")
    prediccion = modelo.predict(datos[0]) 
    fscore=score(datos[1], prediccion,average='macro')
    funciones.estats(datos[1], prediccion)
    print(" Gausiano: F-1 ",fscore[0])
    return fscore[0]

def Bernoulli(datos,modelo):
    #Bernoulli 
    print("--------------------------->Bernoulli<---------------------------")
    prediccion = modelo.predict(datos[0]) 
    fscore=score(datos[1], prediccion,average='macro')
    funciones.estats(datos[1], prediccion)
    print(" Bernoulli: F-1 ",fscore[0])
    return fscore[0]
    
#deficion de main
def main():
    #path = sys.argv[1]
    #naive=sys.argv[2]
    #nombre = sys.argv[3]

    naive=1
    #path = './DATA/completo_train_synth_dengue.csv'
    #path = './DATA/clinica_train_synth_dengue.csv'
    path = './DATA/laboratorio_train_synth_dengue.csv'
    #path = './DATA/completo_train_synth_dengue.csv'
    nombre="./Archivos_salida/laboratorio_Categorical"

    datos = funciones.cargarDatos(path)
    procesado=funciones.procesarDatosNormalizados(datos,2)
    modelo=funciones.cargarBinario(nombre+".nb")
    bandera=True
    if naive==1:#Categorical
        opcion=Categorical(procesado,modelo)
    elif naive==2:#Bernoulli
        opcion=Bernoulli(procesado,modelo)
    elif naive==3:#Gausiano
        opcion=Gausiano(procesado,modelo)
    else:
        print("Opcion Invalida.")
        bandera=False
   

#solo inicia si es el proceso inicial#
if __name__ == "__main__":
    main()