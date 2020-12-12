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
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
#metricas
from sklearn.metrics import precision_recall_fscore_support as score

def Categorical(tempsX,tempsY,tempsPred,TempsVal):
    #Categorical
    #print("--------------------------->Categorical<---------------------------")
    naiveCate= CategoricalNB()
    naiveCate.fit(tempsX.abs(), tempsY)
    prediccion = naiveCate.predict(tempsPred.abs()) 
    fscore=score(TempsVal, prediccion,average='macro')
    #funciones.estats(TempsVal, prediccion)
    #print(" Categorical: F-1 ",fscore[0])
    return fscore[0]

def Gausiano(tempsX,tempsY,tempsPred,TempsVal):
    #Gausiano
    #print("--------------------------->Gausiano<---------------------------")
    naiveGausiano = GaussianNB()
    naiveGausiano.fit(tempsX, tempsY)
    prediccion = naiveGausiano.predict(tempsPred) 
    fscore=score(TempsVal, prediccion,average='macro')
    #funciones.estats(TempsVal, prediccion)
    #print(" Gausiano: F-1 ",fscore[0])
    return fscore[0]

def Bernoulli(tempsX,tempsY,tempsPred,TempsVal):
    #Bernoulli 
    #print("--------------------------->Bernoulli<---------------------------")
    naiveBerno = BernoulliNB()
    naiveBerno.fit(tempsX, tempsY)
    prediccion = naiveBerno.predict(tempsPred) 
    fscore=score(TempsVal, prediccion,average='macro')
    #funciones.estats(TempsVal, prediccion)
    #print(" Bernoulli: F-1 ",fscore[0])
    return fscore[0]

def generarStats(Crosssets):
    tempsX=Crosssets[0]
    tempsY=Crosssets[1]
    tempsPred=Crosssets[2]
    TempsVal=Crosssets[3]
    F1Gausiano=[]
    F1Bernoulli=[]
    F1Categorical=[]
    fd = open('./Estadisticas/salidaNaiveLAB.csv','a') #Salida de configuraciones
    fd.write('Tipo,P1,P2,P3,P4,P5\n')
    for j in range (5):
        F1Gausiano=[]
        F1Bernoulli=[]
        F1Categorical=[]
        for p in range(5):
            train=[]
            test=[]
            for i in range(5):
                if(i!=j):
                    train = pd.concat([tempsX[i], tempsPred[i]])
                    test = np.concatenate((tempsY[i], TempsVal[i]), axis=0)
            F1Categorical.append(Categorical(train,test,tempsPred[j],TempsVal[j]))
            F1Gausiano.append(Gausiano(train,test,tempsPred[j],TempsVal[j]))
            F1Bernoulli.append(Bernoulli(train,test,tempsPred[j],TempsVal[j]))
        linea="Bernoulli,"+str(p)+" "+str(F1Gausiano).strip('[]')+'\n'
        linea2="Gausiano,"+str(p)+" "+str(F1Categorical).strip('[]')+'\n'
        linea3="Categorical,"+str(p)+" "+str(F1Bernoulli).strip('[]')+'\n'
        fd.write(linea)
        fd.write(linea2)
        fd.write(linea3)
    fd.write("\n")
    fd.close() 
    print("--> Escritura exitosa. Datos de analisis generados en GRAFICAS/salidaNaive.csv")
#deficion de main#

def main():
    path = sys.argv[1]
    path = './DATA/laboratorio_train_synth_dengue.csv'
    datos = funciones.cargarDatos(path)
    procesado=funciones.procesarDatosNormalizados(datos,1)
    tags=funciones.getTags(datos,1)
    generarStats(procesado)
    


#solo inicia si es el proceso inicial#
if __name__ == "__main__":
    main()