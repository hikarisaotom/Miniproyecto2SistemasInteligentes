# --------------> Autores <-----------------
#       Claudia Cortés          11711357
#       Ingrid Dominguez        11711355
#

# --------------> Autores <-----------------
#       Claudia Cortés          11711357
#       Ingrid Dominguez        11711355
#

#Para evaluacion de resultados 
from sklearn.metrics import confusion_matrix
#Para procesamiento de datos y demas.
import funcionesUtiles as funciones
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
    for ind in confs.index: #
        kern=str(confs['kernel'][ind])
        c=float(confs['C'][ind])
        Gam=str(confs['Gamma'][ind])
        # Creacion del bosque
        model = SVC( C = c,kernel = kern,gamma = Gam)
        F1Temps=[]
        print("Generando SVC para configuracion ",ind)
        for j in range(5):
            for i in range(5): 
                model.fit(tempsX[i],tempsY[i])
                #Predicciones y metricas
                prediccion = model.predict(tempsPred[i]) 
                fscore=score(TempsVal[j], prediccion,average='macro')
                F1Temps.append(fscore[2])
            linea=kern+','+str(c)+','+Gam+','+','+str(F1Temps).strip('[]')+'\n'
            F1Temps=[]
            fd.write(linea)
        fd.write("\n")
    fd.close()
    print("--> Escritura exitosa. Datos de analisis generados en GRAFICAS/salidaSVC.csv")
        


def main():
    #path = sys.argv[1]
    path = './DATA/laboratorio_train_synth_dengue.csv'
    datos = funciones.cargarDatos(path)
    procesado=funciones.procesarDatosNormalizados(datos)
    entrenar(procesado)

if __name__ == '__main__':
    main()
    
                            


