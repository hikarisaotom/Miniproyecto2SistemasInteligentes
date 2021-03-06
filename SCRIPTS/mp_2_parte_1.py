# --------------> Autores <-----------------
#       Claudia Cortés          11711357
#       Ingrid Dominguez        11711355
#
#librerias
import sys
#Para Manejo de archivos
import pandas as pd
from pandas import ExcelWriter
#para Generar graficas
import matplotlib.pyplot as plt
#Para procesamiento de datos y demas.
import funcionesUtiles as funciones

def generar(datos):
    #Lista de attributos discreto.
    atributosContinuos = ["plaquetas","linfocitos","hematocritos","leucocitos"]
    atributos=list(datos.columns.values)
    atributos.remove("clase") 
    #Se inicializa el lector del archivo
    writer = pd.ExcelWriter('./Estadisticas/estadisticas.xlsx', engine = 'xlsxwriter')
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
                nombre="./Estadisticas/"+atributo+".png"
                plt.savefig(nombre)
                plt.clf()
    print("Finalizado, revise la carpeta Estadisticas")
    writer.save()

#deficion de main#
def main():
    #path = sys.argv[1]
    path = './DATA/completo_train_synth_dengue.csv'
    datos = pd.read_csv(path, engine='python')
    datos=funciones.preprocesar(datos)
    generar(datos)

#solo inicia si es el proceso inicial#
if __name__ == "__main__":
    main()