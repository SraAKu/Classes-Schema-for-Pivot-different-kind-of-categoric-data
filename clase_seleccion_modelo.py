import pandas as pd
import numpy as np
import os 
from scipy import stats
from itertools import permutations
import datetime

from datetime import datetime as dt
import datetime as dtt
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV,cross_val_score

from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn import FunctionSampler
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import make_pipeline

from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline as imbpipeline

from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay

class Seleccionar_Modelo():
    '''
    La clase compara modelos y ajusta los parámetros e hiperpárametros de los 
    modelos de clasificación.
    Parametros:

        X (Dataframe): Dataframe de catacteristicas

        y (Dataframe): Etiquetas

        test_size (float | valor): Numero entre 0.0 y 1.0 representa la 
        proporciòn del conjunto de datos para incluir en los datos test.
        Se dividen la base de datos original en datos de test y train,
        utilizamos los datos de train para ajustar los paràmetros e 
        hiperparàmetros y los datos test para seleccionar el mejor modelo.
        Por defecto el valor es 0.2

        random_state (int | valor): int para una salida reproducible

        kfold_n_split (int | valor): int Number of folds,por defecto 5: 
        se divide en 5 grupos en los datos de train y un grupo se selecciona
        para validar, esto se repite en todos los grupos

        kfold_n_repeats (int | valor): Numero de veses que es necesario repetir
        el validador cruzado (El paràmetro anterior)

        indx_model (int | valor): Me permite seleccionar el mejor modelo de los modelos
        que ingresò, es un indice, el valor por defecto es None
        
        n_jobs (int | valor): Variable auxiliar

        scoring (str | valor): Es la metrica que se utiliza para seleccionar el mejor
        modelo,  se recomienda utilizar F1_score y roc_auc
        
        refit (bool | valor): Variable auxiliar

        equilibrar_clases (bool | valor): bool por defecto true, en caso de que
        las clases estèn equilibradas. Este parametro nos permite equilibrar los
        datos con algunos de los metodos mencionados anteriormente

        random_state_muestreo (int | valor): int para una salida reproducible

        sampling_strategy (float, str | valor): float, str informaciòn de
        muestreo para muestrear el conjunto de datos.

            cuando float, corresponde a la proporciòn deseada del nùmero de muestras
            en la clase minoritaria sobre el numero de muestras en la clase 
            mayoritaria dspuès de el remuestreo

            cuando str, especifique la clase a la que se dirige el remuestreo.
            Se igualara el numero de muestras en las diferentes clases. Las posibles
            opciones son: _'majority'_ remuestrear solo la clase mayoritaria;
            _'not minority'_ remuestrear todas las clases excepto la clase
            minoritaria; _'not majority'_ remuestrea todas las clases excepto la 
            clase mayoritaria; _'all'_ remuestrear todas las clases _'auto'_
            equivalente a 'not minority'

        muestreo (str | valor): list, tuple (tipo de muestreo, metodo)

    '''
    def __init__(self,
                 X, y, 
                 test_size = 0.2, 
                 random_state = None,
                 kfold_n_split = 5,
                 kfold_n_repeats = 2,
                 indx_model = None,
                 n_jobs = -1,
                 scoring = 'roc_auc',
                 refit = True,
                 equilibrar_clases = True,
                 random_state_muestreo = None, 
                 sampling_strategy = 'auto',
                 muestreo = ('sobremuestreo','smote')):
        self.X = X
        self.y = y
        self.test_size       = test_size
        self.random_state    = random_state
        self.kfold_n_split   = kfold_n_split
        self.kfold_n_repeats = kfold_n_repeats
        self.indx_model      = indx_model
        self.n_jobs   = n_jobs
        self.scoring  = scoring
        self.refit    = refit
        self.muestreo = muestreo
        self.equilibrar_clases     = equilibrar_clases
        self.random_state_muestreo = random_state_muestreo
        self.sampling_strategy     = sampling_strategy
    
    def muestro_datos(self):
        '''
        Permite seleccionar el metodo de entrada

        Entradas: Parametros de entrada clase

        Salida: Objeto     
        '''
        if self.muestreo[0] == 'submuestreo':
            
            if self.muestreo[1] == 'random':
                under  = RandomUnderSampler(sampling_strategy = self.sampling_strategy,
                                            random_state = self.random_state_muestreo)
            if self.muestreo[1] == 'cluster':
                under  = ClusterCentroids(sampling_strategy = self.sampling_strategy,
                                          random_state = self.random_state_muestreo)
            if self.muestreo[1] == 'vecinos':
                under  = RepeatedEditedNearestNeighbours(sampling_strategy = self.sampling_strategy,
                                                         random_state = self.random_state_muestreo) 
            if self.muestreo[1] == 'umbral':
                under  = InstanceHardnessThreshold(sampling_strategy = self.sampling_strategy,
                                                    random_state = self.random_state_muestreo) 
        
            return under
        
        ## --------Sobremuestreo-----------
        if self.muestreo[0] == 'sobremuestreo':
            
            if self.muestreo[1] == 'random':
                over  = RandomOverSampler(sampling_strategy = self.sampling_strategy,
                                          random_state = self.random_state_muestreo) 
            if self.muestreo[1] == 'smote':
                over  = SMOTE(sampling_strategy = self.sampling_strategy,
                              random_state = self.random_state_muestreo) 
            if self.muestreo[1] == 'adasyn':
                over  = ADASYN(sampling_strategy = self.sampling_strategy,
                                random_state = self.random_state_muestreo) 
                
            return over

    def selection_best_model(self,list_grid_search, X_test, y_test):
        '''
        Selecciona el mejor modelo

        Entradas: Parametros de entrada clase 
        Grilla de parametros 
        Datos de test

        Salidas: Seleccionar mejor modelo 
                 Seleccionar mejores parametros 

        '''        
        test_score = [model.score(X_test,np.array(y_test).ravel()) for model in list_grid_search]
        
        if self.indx_model == None:
            indx_best_test = np.argmax(test_score)
        else:
            indx_best_test = int(self.indx_model)
        
        self.index_best_model = indx_best_test
        self.best_model  = list_grid_search[indx_best_test].best_estimator_
        self.best_score  = list_grid_search[indx_best_test].best_score_
        self.best_params = list_grid_search[indx_best_test].best_params_
        
        
        self.models  = list_grid_search
        self.scores  = [ii.best_score_ for ii in list_grid_search]
        self.paramss = [ii.best_params_ for ii in list_grid_search]
        
        self.test_score = test_score
            
    def entrenar_modelos(self, pipeline, parametros):
        '''
        Entrena modelos

        Entradas: Parametros de entrada clase
        pipeline
        parametros modelos

        Salidas: Mejor modelo

        '''
        # Dividir los datos en Train y test
        X_train, X_test, y_train, y_test = train_test_split(self.X,
                                                            self.y, 
                                                            stratify  = self.y,
                                                            test_size = self.test_size, 
                                                            random_state = self.random_state)
        
        ## Indeces conjuntos de train y text
        self.index_train, self.index_test =  y_train.index, y_test.index
        
        # Agrega la pipeline el muestreo de los datos
        if self.equilibrar_clases:
            sample     = self.muestro_datos()
            pipeline_a  = [[('sample',sample)] + pp for pp in pipeline]
            pipeline_c = [imbpipeline(pp) for pp in pipeline_a]
            
            print('\nPipeline modelo:')
            for ii,jj in enumerate(pipeline_c):
                print(f'   Modelo {ii+1}:')
                print(f'       {jj}')
            print('\n...')
        else:
            pipeline_c =[ Pipeline(pp) for pp in pipeline]
            
            print('\nPipeline modelo:')
            for ii,jj in enumerate(pipeline_c):
                print(f'   Modelo {ii+1}:')
                print(f'       {jj}')
            print('\n...')
            
        if isinstance(self.scoring, list) or isinstance(self.scoring, tuple):
            aux_refit =  self.scoring[0]
        else:
            aux_refit =  self.refit

            
        # Validacion cruzada    
        kfold            = RepeatedStratifiedKFold(n_splits  = self.kfold_n_split, 
                                                   n_repeats = self.kfold_n_repeats)
        list_grid_search = []
        
        #  Selecciona los mejores parametros para cada uno de los modelos evaluar
        for pp,qq in zip(pipeline_c,parametros):
            grid_search = GridSearchCV(pp, qq, 
                                       n_jobs = self.n_jobs,
                                       cv = kfold, 
                                       #refit = aux_refit,
                                       scoring = self.scoring,
                                       verbose = 10)

            ##------------Entrena modelo----------------------
            # Entrenar modelo con lo mejores parametros
            grid_search.fit(X_train, np.array(y_train).ravel())
            list_grid_search.append(grid_search)
            
        # Evaluar modelos con los mejores parametros en conjunto de Test
        self.selection_best_model(list_grid_search, X_test, y_test)
        
    def fit(self,pipeline,parametros,*_):
        '''
        Ajustar datos al modelo

        Entradas: Parametros de entrada clase
        pipeline
        parametros modelos

        salidas: Mejor modelo

        '''
        self.entrenar_modelos(pipeline, parametros)
        return self