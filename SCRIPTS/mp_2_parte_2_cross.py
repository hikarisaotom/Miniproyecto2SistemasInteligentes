# --------------> Autores <-----------------
#       Claudia Cortés          11711357
#       Ingrid Dominguez        11711355
#
#librerias
import sys
import numpy as np
#Para Manejo de archivos
import csv
#Random forest 
from sklearn.ensemble import RandomForestClassifier
#Evaluar el rendimiento 
from sklearn.metrics import precision_recall_fscore_support as score
#Para procesamiento de datos y demas.
import funcionesUtiles as funciones

def entrenar(Crosssets):
    tempsX=Crosssets[0]
    tempsY=Crosssets[1]
    tempsPred=Crosssets[2]
    TempsVal=Crosssets[3]
    F1Temps=[]
    confs = funciones.cargarDatos('./confs/configuraciones.csv') #Cargar configuraciones
    fd = open('./Estadisticas/Salida.csv','a') #Salida de configuraciones
    fd.write('Criterio,Arboles,Profundidad,Atributos\n')
    for ind in confs.index: 
        criterio=str(confs['criterio'][ind])
        arboles=int(confs['arboles'][ind])
        profundidad=int(confs['profundidad'][ind])
        atributos=str(confs['atributos'][ind])
        # Creacion del bosque
        bosque = RandomForestClassifier(criterion=criterio,n_estimators=arboles,max_depth=profundidad,max_features=atributos)
        F1Temps=[]
        print("Generando Random Forest para configuracion ",ind)
        for j in range(5):
            for i in range(5): 
                bosque.fit(tempsX[i],tempsY[i])
                #Predicciones y metricas
                prediccion = bosque.predict(tempsPred[i]) 
                fscore=score(TempsVal[j], prediccion,average='macro')
                F1Temps.append(fscore[2])
            linea=criterio+','+str(arboles)+','+str(profundidad)+','+atributos+','+str(F1Temps).strip('[]')+'\n'
            F1Temps=[]
            fd.write(linea)
        fd.write("\n")
    fd.close() 
    print("--> Escritura exitosa. Datos de analisis generados en GRAFICAS/salida.csv")


#deficion de main#
def main():
    #path = sys.argv[1]
    path = './DATA/completo_train_synth_dengue.csv'
    datos = funciones.cargarDatos(path)
    procesado=funciones.procesarDatos(datos,1)
    entrenar(procesado)

#solo inicia si es el proceso inicial#
if __name__ == "__main__":
    main()
