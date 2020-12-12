#Manejo de arreglos
import numpy as np
#Manejo de archivos
import pandas as pd
import pickle
#Para crear los sets del crossvalidation
from sklearn.model_selection import train_test_split
#Para estadisticas
from sklearn.metrics import classification_report, confusion_matrix
#Para normalizar
from sklearn.preprocessing import StandardScaler
def preprocesar(datos):
    datos=datos.replace(np.nan,2,regex=True)
    datos=datos.replace("NO","No",regex=True)
    return datos


def oneHot(datos):
    #One hot encoding
    datos = pd.get_dummies (datos)
    # Convertir a características de matriz 
    numpy = np.array (datos)
    return datos

def getTags(datos,n ):
    #Obtenemos los valores a predecir
    if n ==2:
        datos = datos.replace("Dengue_Grave", "0", regex=True)
        datos = datos.replace("Dengue_NoGrave_NoSignos", "1", regex=True)
        datos = datos.replace("Dengue_NoGrave_SignosAlarma", "2", regex=True)
        datos = datos.replace("No_Dengue", "3", regex=True)
    tags = np.array (datos ['clase'])
    return tags

def limpiarFeatures(datos):
    # Elimina las etiquetas de las columnas
    datos = datos.drop ('clase', axis = 1)
    return datos

def separar(datos,tags):
    #Sets para CrossValidation
    entrenamiento1, prueba1, entrenamiento2, prueba2 = train_test_split(datos, tags, test_size=0.50, random_state=66,shuffle=True) #dejamos el 50%  para prueba
    #segunda ronda de sets
    entrenamiento3, prueba3, entrenamiento4, prueba4 = train_test_split(datos, tags, test_size=0.50, random_state=70,shuffle=True) #dejamos el 50%  para prueba
    #Tercera ronda.
    entrenamiento5, prueba5,entrenamiento6, prueba6 = train_test_split(datos, tags, test_size=0.50, random_state=1,shuffle=True) #dejamos el 50%  para prueba
    #Cuarta ronda.
    entrenamiento7, prueba7,entrenamiento8, prueba8 = train_test_split(datos, tags, test_size=0.50, random_state=1,shuffle=True) #dejamos el 50%  para prueba
    #Quinta ronda.
    entrenamiento9, prueba9,entrenamiento10, prueba10 = train_test_split(datos, tags, test_size=0.50, random_state=1,shuffle=True) #dejamos el 50%  para prueba
    tempsX=[entrenamiento1,entrenamiento3,entrenamiento5,entrenamiento7,entrenamiento9]
    tempsY=[entrenamiento2,entrenamiento4,entrenamiento6,entrenamiento8,entrenamiento10]
    tempsPred=[prueba1,prueba3,prueba5,prueba7,prueba9]
    TempsVal=[prueba2,prueba4,prueba6,prueba8,prueba10]
    Respuesta=[tempsX,tempsY,tempsPred,TempsVal]
    return Respuesta

def separarCompleto(datos,tags):
     #Sets para CrossValidation
    entrenamiento1, prueba1, entrenamiento2, prueba2 = train_test_split(datos, tags, test_size=0.0000001, random_state=66,shuffle=True) 
    #uniendolos 
    train = pd.concat([entrenamiento1, prueba1])
    test = np.concatenate((entrenamiento2, prueba2), axis=0)
    return  train, test



def procesarDatos(datos,op):
    datos=preprocesar(datos)
    # Valores a predecir 
    tags=getTags(datos,1)
    #Limpiando y eliminando columnas de resultados
    datos=limpiarFeatures(datos)
    #one hot encoding de los demas attrs
    datos=oneHot(datos)
    if op==1:
        return separar(datos,tags)
    else: 
        return separarCompleto(datos,tags)

def normalizar(datos):
    scaler = StandardScaler()
    transformados = scaler.fit_transform(datos)
    datos = pd.DataFrame(transformados)
    return datos

def procesarDatosNormalizados(datos, op):
    datos=preprocesar(datos)
    # Valores a predecir 
    tags=getTags(datos,2)
    #Limpiando y eliminando columnas de resultados
    datos=limpiarFeatures(datos)
    #one hot encoding de los demas attrs
    datos=oneHot(datos)
    #Normalizando datos
    datos=normalizar(datos)
    if op==1:
        return separar(datos,tags)
    else: 
        return separarCompleto(datos,tags)






def cargarDatos(path):
    return pd.read_csv(path,engine='python')


def GuardarBinario(objeto,ruta):
    archivo = open(ruta, 'wb')
    pickle.dump(objeto,archivo)
    archivo.close()
    print(ruta+" Fue salvado exitosamente")
    

def cargarBinario(ruta):
    with open(ruta, 'rb') as archivo:
        objeto = pickle.load(archivo)
    archivo.close()
    return objeto

def estats(tags, prediccion):
    print("------>ESTADISTICAS<------")
    print(classification_report(tags, prediccion))
    print("------>MATRIZ CONFUSION<------")
    labels=["Dengue_Grave","Dengue_NoGrave_NoSignos","Dengue_NoGrave_SignosAlarma","No_Dengue"]
    vals = confusion_matrix(tags, prediccion)
    print('{0:50} {1:8} {2:8} {3:8} {4:8}'.format(" clase ","   tn", "   fp", "     fn","    tp"))
    for i in range(4):
        val=vals[i]
        print('{0:50} {1:8} {2:8} {3:8} {4:8}'.format(labels[i],val[0], val[1], val[2], val[3]))
