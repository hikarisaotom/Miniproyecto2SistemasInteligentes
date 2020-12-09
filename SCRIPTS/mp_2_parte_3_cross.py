# --------------> Autores <-----------------
#       Claudia Cortés          11711357
#       Ingrid Dominguez        11711355
#

# --------------> Autores <-----------------
#       Claudia Cortés          11711357
#       Ingrid Dominguez        11711355
#

import utilities as util
from sklearn.svm import SVC as model

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

def main():
    
    data_sets = ["clinica_train_synth_dengue.csv",
                    "laboratorio_train_synth_dengue.csv",
                    "completo_train_synth_dengue.csv"]
    
    trees = [ i*10 for i in range(5) ]    
    criterion = ['gini','entropy']
    max_depth = [None,3,6]
    max_features = ['auto', 'log2']
    groups = 5

    for a in data_sets:
        X,Y = util.process_data(a)
        for b in trees:
            for c in criterion:
                for d in max_depth:
                    for e in max_features:                 
                        model = model(
                            n_estimators = b,
                            criterion = c,
                            max_depth = d,
                            max_features = e
                        )
                        groups_X,groups_Y = util.split_data(groups,X,Y)
                        for i in range(groups):
                            validation_X = groups_X[i]
                            validation_Y = groups_Y[i]
                            for j in range(groups):
                                if j != i:
                                    training_X += groups_X[i]
                                    training_Y += groups_Y[i]                            
                            model.fit(training_X,training_Y)
                            F1_val = util.F1(model,validation_X,validation_Y)
                            print(util.print_data(a,b,c,d,e,i,F1_val))
                            


