 # --------------> Autores <-----------------
#       Claudia Cortés          11711357
#       Ingrid Dominguez        11711355
#

  #  cv = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)
   # total=  zip(cv.split(datos, tags), range(5))
 
    #for (train, test), i in zip(cv.split(datos, tags), range(5)):
     #   nuevo=datos.iloc[test]
      #  print("traing shape",nuevo.shape)
       # print("test shape",train.shape)
        #bosque.fit(nuevo, test)
        #prediccion = bosque.predict(nuevo) 
        #print("------>ESTADISTICAS PARAEL RANDOM FOREST<------")
        #print(classification_report(test, prediccion))




 for conf in range(15):
        criterio='entropy'
        arboles=200
        profundidad=15
        atributos=17
        # Creacion del bosque
        bosque = RandomForestClassifier(criterion=criterio,
            n_estimators=arboles,
            max_depth=profundidad,
            max_features=atributos)
        for i in range(5): 
                bosque.fit(tempsX[i],tempsY[i])
                #Predicciones y metricas
                prediccion = bosque.predict(tempsPred[i]) 
                print("------>ESTADISTICAS PARAEL RANDOM FOREST<------")
                print(classification_report(TempsVal[i], prediccion))
                precision,recall,fscore,support=score(TempsVal[i], prediccion,average='macro')
                print('F-score   : {}'.format(fscore))
                print('\n')
                F1Temps.append(fscore)
 