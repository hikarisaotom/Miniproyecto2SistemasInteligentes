# --------------> Autores <-----------------
#       Claudia Cortés          11711357
#       Ingrid Dominguez        11711355
#
#librerias
import sys
import numpy as np
#Random forest 
from sklearn.ensemble import RandomForestClassifier
#Para procesamiento de datos y demas.
import funcionesUtiles as funciones

def entrenar(Datos):
    X=Datos[0]
    Y=Datos[1]
    bosque = RandomForestClassifier(
        criterion='gini',
        n_estimators=154,
        max_depth=21,
        max_features='auto')
        		
    return bosque.fit(X,Y)

#deficion de main
def main():
    path = sys.argv[1]
    nombre = sys.argv[2]
    #path = './DATA/completo_train_synth_dengue.csv'
    #path = './DATA/clinica_train_synth_dengue.csv'
    #path = './DATA/laboratorio_train_synth_dengue.csv'
    #path = './DATA/completo_train_synth_dengue.csv'
    #nombre="./Archivos_salida/laboratorio"
    datos = funciones.cargarDatos(path)
    procesado=funciones.procesarDatos(datos,2)
    bosque=entrenar(procesado)
    print("Bosque Entrenado Exitosamente")
    funciones.GuardarBinario(bosque,nombre+".rfc")

#solo inicia si es el proceso inicial#
if __name__ == "__main__":
    main()