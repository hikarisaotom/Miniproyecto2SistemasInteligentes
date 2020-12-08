# --------------> Autores <-----------------
#       Claudia Cortés          11711357
#       Ingrid Dominguez        11711355
#
#librerias#
import sys
from time import time
import numpy as np
#manejo de archivos
import pandas as pd
from pandas import ExcelWriter
#graficas
import matplotlib.pyplot as plt
#sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, mean_squared_error, r2_score

def tablas(datos):
    #Lista de attributos discreto.
    atributosDiscretos = ['sexo','dias_fiebre','dias_ultima_fiebre','nauseas','rash','vomitos','mialgias','artralgias','dolor_abdominal','acumulacion_fluidos','sangrado_mucosas','hemorragia','letargia','irritabilidad','shock','prueba_torniquete','hepatomegalia']
    #Se inicializa el lector del archivo
    writer = pd.ExcelWriter('./GRAFICAS/estadisticas.xlsx', engine = 'xlsxwriter')
    for atributo in atributosDiscretos: 
        print("Generando ",atributo)
        Cajas=[]
        filtrado = datos[[atributo, "clase"]]
        valores=pd.unique(datos[atributo]).tolist()
        temp1=[]
        temp3=[]
        temp2=[]
        temp4=[]
        for opcion in valores:
            box=[]
            subSet =filtrado[filtrado[atributo] == opcion]
            temp1.append(subSet[subSet['clase'] == 'No_Dengue'].shape[0])
            temp2.append(subSet[subSet['clase'] == 'Dengue_NoGrave_NoSignos'].shape[0])
            temp3.append(subSet[subSet['clase'] == 'Dengue_NoGrave_SignosAlarma'].shape[0])
            temp4.append(subSet[subSet['clase'] == 'Dengue_Grave'].shape[0])
            size=len(temp1)-1
            box=[temp1[size],temp2[size],temp3[size],temp4[size]]
            Cajas.append(box)
        #Generando Archivo
        archivo= pd.DataFrame({ atributo: valores,
                    'No Dengue': temp1,
                    'NNo signos Alerta': temp2,
                    'Signos Alerta': temp3,
                    'Dengue  Grave': temp4})
        archivo.to_excel(writer, sheet_name=atributo,index=False)
        #Generando Grafica
        plt.boxplot(Cajas)
        labels = valores
        plt.xticks(np.arange(len(labels))+1,labels)
        plt.title(atributo)
        nombre="./GRAFICAS/"+atributo+".png"
        plt.savefig(nombre)
        #plt.show()

    print("Finalizado, revise la carpeta GRAFICAS")
    writer.save()

#deficion de main#
def main():
   #archivo1 = sys.argv[1]
    archivo1 = './DATA/clinica_train_synth_dengue.csv'
    archivo2 = './DATA/laboratorio_train_synth_dengue.csv'
    archivo3 = './DATA/completo_train_synth_dengue.csv'
    datos1 = pd.read_csv(archivo1, engine='python')
    datos2 = pd.read_csv(archivo2, engine='python')
    datos3 = pd.read_csv(archivo3, engine='python')
    #para ver informacion
    print("DATOS DE CLINICA")
    tablas(datos1)
   # print("-------------------------------------------------------------------")
    #print("DATOS DE LAB")
    #tablas(datos2)
    #print("-------------------------------------------------------------------")
    #print("DATOS DE CLINICA Y LAB")
    #tablas(datos3)

#solo inicia si es el proceso inicial#
if __name__ == "__main__":
    main()

    #random forest
    #MSVMaquinas Soporte Vecto
    #Bayesiano ingenuo

    #Clases:
    # No dengue
    # Dengue no grave y sin isgos de alarma
    #Dengue no grave con signos de alarma
    #Dengue Grave