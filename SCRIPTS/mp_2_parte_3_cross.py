# --------------> Autores <-----------------
#       Claudia Cortés          11711357
#       Ingrid Dominguez        11711355
#

import sys
#Para procesamiento de datos y demas.
import funcionesUtiles as funciones
import pandas as pd
import numpy as np
#Para SVC
from sklearn.svm import SVC as SVC
#metricas
from sklearn.metrics import precision_recall_fscore_support as score

def entrenar(Crosssets):
    tempsX=Crosssets[0]
    tempsY=Crosssets[1]
    tempsPred=Crosssets[2]
    TempsVal=Crosssets[3]
    F1Temps=[]
    confs = funciones.cargarDatos('./confs/configuracionesSVC.csv') #Cargar configuraciones
    fd = open('./Estadisticas/SalidaSVC.csv','a') #Salida de configuraciones
    fd.write('kernel,C,Gamma\n')
    for ind in confs.index: 
        kern=str(confs['kernel'][ind])
        c=float(confs['C'][ind])
        Gam=str(confs['Gamma'][ind])
        # Creacion del SVC
        model = SVC( C = c,kernel = kern,gamma = Gam)
        print("Generando SVC para configuracion ",ind)
        for j in range(5):
            F1Temps=[]
            for p in range(5):
                X=[]
                Y=[]
                for i in range(5): 
                    if(i!=j):
                        X = pd.concat([tempsX[i], tempsPred[i]])
                        Y = np.concatenate((tempsY[i], TempsVal[i]), axis=0)
                model.fit(tempsX[i],tempsY[i])
                #Predicciones y metricas
                prediccion = model.predict(tempsPred[i]) 
                fscore=score(TempsVal[j], prediccion,average='macro')
                F1Temps.append(fscore[2])
            linea=kern+','+str(c)+','+Gam+','+','+str(F1Temps).strip('[]')+'\n'
            fd.write(linea)
        fd.write("\n")
    fd.close()
    print("--> Escritura exitosa. Datos de analisis generados en Estadisticas/salidaSVC.csv")
        


def main():
    path = sys.argv[1]
    datos = funciones.cargarDatos(path)
    procesado=funciones.procesarDatosNormalizados(datos,1)
    entrenar(procesado)
    #clinica_train_synth_dengue
    #laboratorio_train_synth_dengue
    #completo_train_synth_dengue
if __name__ == '__main__':
    main()
    
                            


