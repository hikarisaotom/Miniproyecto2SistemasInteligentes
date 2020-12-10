# --------------> Autores <-----------------
#       Claudia Cortés          11711357
#       Ingrid Dominguez        11711355
#
#librerias
#Otras funciones
import sys
import numpy as np
#Para Manejo de archivos
import pandas as pd
from pandas import ExcelWriter
#para Generar graficas
import matplotlib.pyplot as plt

def preprocesar(datos):
    datos=datos.replace(np.nan,'NA',regex=True)
    datos=datos.replace("NO","No",regex=True)
    return datos


def generar(datos):
    #Lista de attributos discreto.
    atributosContinuos = ["plaquetas","linfocitos","hematocritos","leucocitos"]
    atributos=list(datos.columns.values)
    atributos.remove("clase") 
    #Se inicializa el lector del archivo
    writer = pd.ExcelWriter('./GRAFICAS/estadisticas.xlsx', engine = 'xlsxwriter')
    for atributo in atributos: 
        filtrado = datos[[atributo, "clase"]]
        temp1=[]
        temp3=[]
        temp2=[]
        temp4=[]
        valores=pd.unique(datos[atributo]).tolist()
        if not (atributo in atributosContinuos):#Generar tabla
            for opcion in valores:
                subSet =filtrado[filtrado[atributo] == opcion]
                temp1.append(subSet[subSet['clase'] == 'No_Dengue'].shape[0])
                temp2.append(subSet[subSet['clase'] == 'Dengue_NoGrave_NoSignos'].shape[0])
                temp3.append(subSet[subSet['clase'] == 'Dengue_NoGrave_SignosAlarma'].shape[0])
                temp4.append(subSet[subSet['clase'] == 'Dengue_Grave'].shape[0])
            archivo= pd.DataFrame({ atributo: valores,
                        'No Dengue': temp1,
                        'No signos Alerta': temp2,
                        'Signos Alerta': temp3,
                        'Dengue  Grave': temp4})
            archivo.to_excel(writer, sheet_name=atributo,index=False)
        elif atributo in atributosContinuos: #Generar Grafica
                plt.boxplot(x=datos[atributo],data= filtrado)
                plt.title(atributo)
                nombre="./GRAFICAS/"+atributo+".png"
                plt.savefig(nombre)
                plt.clf()
    print("Finalizado, revise la carpeta GRAFICAS")
    writer.save()

#deficion de main#
def main():
    path = sys.argv[1]
    datos = pd.read_csv(path, engine='python')
    datos=preprocesar(datos)
    generar(datos)

#solo inicia si es el proceso inicial#
if __name__ == "__main__":
    main()