import pandas as pd
import numpy as np
import os 
from scipy import stats
from itertools import permutations
import datetime

from datetime import datetime as dt
import datetime as dtt
from dateutil.relativedelta import relativedelta, MO
from sklearn.linear_model import LinearRegression

import locale
locale.setlocale(locale.LC_ALL,'es_ES.UTF-8')
from datetime import datetime as dt

class Caracterizacion_Clientes():  
    '''
    Clase de caracterización de clientes de acuerdo a los parametros 
    especificados 
    Parametros:

        producto ([str] | lista): Nombre  de producto/s de la base de datos

        delta_t  ((str, int) | tupla): Rango para definir cliente perdido en 
        'día', 'semana' o 'mes'

        lista_caract ([str] | lista): Lista de caracteristicas que se tendrán en
        cuenta

        num_transacciones_minimo (int | valor): Cantidad de transacciones minima
        en productos para considerarse cliente

        tiempo_pendiente (str | valor): Detalle para el calculo de pendientes
        default = 'mes'

        rango_tiempo ((str, int) | tupla): Rango de tiempo para el análisis
        default = None

        dias_entre_semana ([object] | lista): lista de nombres o valores para 
        los días de la semana
        default = [1.0, 2.0, 3.0, 4.0, 5.0]

        dias_fin_semana ([object] | lista): lista de nombres o valores para los 
        días de fin de semana
        default = [6.0, 7.0]

        all_transacciones (bool | valor): True si se cuentan todas las 
        transacciones - False si se tienen en cuenta solo transacciones por hora
        default = True

        agregacion_caract (dict): diccionario especificando el nivel de agregación
        de cada caracteristica que se tendrá en cuenta, incluyendo nombre y 
        seudonimo de la caracteristica {'seudonimo': ('agregación', 'variable')}.
            'agregacion': -'moda' -'moda_dummie' -'conteo' -'suma_tx' -'suma_vlr' 
                          -'suma_tx_tiempo' -'suma_vlr_tiempo'
        Si la agregación es nula la caracteristica es de tipo especial y se calcula
        con un metodo especificado al interior de la clase.
    '''

    def __init__(self, producto,
                 delta_t, 
                 num_transacciones_minimo,
                 tiempo_pendiente = 'mes',
                 rango_tiempo = None,
                 dias_entre_semana = [1.0, 2.0, 3.0, 4.0, 5.0],
                 dias_fin_semana   = [6.0, 7.0],
                 all_transacciones = True,
                 agregacion_caract = dict()):
        
        self.producto     = producto
        self.rango_tiempo = rango_tiempo
        self.periodo_t    = delta_t[0]
        self.valor_t      = delta_t[1]
        self.tiempo_pendiente  = tiempo_pendiente
        self.dias_entre_semana = dias_entre_semana
        self.dias_fin_semana   = dias_fin_semana
        self.all_transacciones        = all_transacciones
        self.num_transacciones_minimo = num_transacciones_minimo
        self.agregacion_caract = agregacion_caract

    def eliminar_nombres_repetidos(self,lista):
        '''Elimina valores repetidos de la lista'''
        lista = list(map(str, lista))
        perm = permutations(lista,2)
        for ii,jj in list(perm):
            if ii in jj and ii in lista:
                lista.remove(ii)
        return lista  

    def R_Class(self,x,p,d):
        '''Función de puntaje de Recencia'''
        if x <= d[p][0.2]:
            return 5
        elif x <= d[p][0.4]:
            return 4
        elif x <= d[p][0.6]: 
            return 3
        elif x <= d[p][0.8]: 
            return 2
        elif np.isnan(x): 
            return np.nan
        else:
            return 1

    def FM_Class(self,x,p,d):
        '''Función de puntaje de Frecuencia y valor Monetario'''
        if x <= d[p][0.2]:
            return 1
        elif x <= d[p][0.4]:
            return 2
        elif x <= d[p][0.6]: 
            return 3
        elif x <= d[p][0.8]: 
            return 4
        elif np.isnan(x): 
            return np.nan
        else:
            return 5

    def agg_moda(self, data_producto, variable):
        '''Calcula por cliente la moda de la variable seleccionada en cuanto a 
        sus transacciones'''
        return pd.DataFrame(data_producto.groupby(['cliente_documento'])
                            [variable].apply(lambda x:  stats.mode(x).mode[0]))

    def agg_moda_dummie(self, data_producto, variable):
        '''Calcula por cliente la moda de la variable seleccionada en cuanto a 
        sus transacciones y convierte la variable en dummie'''
        df_variable = self.agg_moda(data_producto, variable)
        #df_variable.columns = [variable + ii for ii in df_variable.columns.tolist()]
        return pd.get_dummies(df_variable)#, drop_first = True)

    def agg_transacciones_pivote(self, data_producto, variable):
        '''Cuenta por cliente el número de fechas en que se realizaron 
        transacciones por categoría dentro de la variable'''
        df_variable = pd.DataFrame(data_producto.groupby(['cliente_documento']
                                                   )[variable].apply(
                                                    lambda x:  list(x)))
        var_list = data_producto[variable].unique().tolist()
        var_list = self.eliminar_nombres_repetidos(var_list)
        
        for ii in var_list:
            df_variable[ii] = df_variable[variable].apply(lambda x: x.count(ii))

        df_variable.columns = [variable + ii for ii in df_variable.columns.tolist()] 

        df_variable = df_variable.drop(columns = [variable+variable])
        return df_variable

    def agg_val_trans_suma(self, data_producto, val_trans, variable):
        '''Condensa valores o transacciones por mes, por cliente'''
        df_val_trans = pd.DataFrame(data_producto.groupby(
                                ['cliente_documento',variable])
                                [val_trans].apply(lambda x: sum(list(x))))
                                
        df_val_trans = df_val_trans.reset_index()
        df_val_trans = df_val_trans.groupby(
                                ['cliente_documento', variable]
                                ,as_index = False).sum().pivot(
                                'cliente_documento',variable).fillna(0)
        df_val_trans.columns = df_val_trans.columns.get_level_values(1)

        df_val_trans.columns = [variable + str(ii) for ii in df_val_trans.columns.tolist()] 
        #df_val_trans = df_val_trans.drop(columns = [variable+variable])
  
        return df_val_trans


    def agg_val_trans_suma_tiempo(self, data_producto, val_trans, rango_tiempo):
        '''Condensa valores o transacciones por mes, por cliente'''
        df_val_trans_mes = pd.DataFrame(data_producto.groupby(
                                ['cliente_documento',rango_tiempo])
                                [val_trans].apply(lambda x: sum(list(x))))

        df_val_trans_mes = df_val_trans_mes.reset_index()
        df_val_trans_mes = df_val_trans_mes.groupby(
                                ['cliente_documento',rango_tiempo]
                                ,as_index = False).sum().pivot(
                                'cliente_documento',rango_tiempo).fillna(0)
        df_val_trans_mes.columns = df_val_trans_mes.columns.get_level_values(1)

        if val_trans == 'vlr_transaccion':
            df_val_trans_mes.columns = ['vlr_transaccion_' + ii 
                                    for ii in df_val_trans_mes.columns.tolist()]     
        else:
             df_val_trans_mes.columns = ['transaccion_' + ii 
                                    for ii in df_val_trans_mes.columns.tolist()]   
        return df_val_trans_mes


    def generar_vector_tiempo(self, vector_tiempo, x):
        '''
        Genera arreglos de valores y cantidad de transacciones para 
        el cálculo de linea de tendencia:
        Entrada:

            vector_tiempo (pandas date range): intervalo de fechas para la 
            regresión. Se calcula en días, semanas o meses.

            x (object | list): lista con arreglos de valores y cantidad de 
            transacciones por fecha para cada cliente.

        Salida:

            (num_Tx_tiempo, vlr_Tx_tiempo) (listas | tupla): tupla de listas con
            valores de transacciones en el tiempo y cantidad de transacciones en 
            el tiempo.
        '''
        num_Tx_tiempo = np.zeros(len(vector_tiempo), dtype = int).tolist()
        vlr_Tx_tiempo = np.zeros(len(vector_tiempo), dtype = int).tolist()
        for ii in np.unique(x[2]):
            if ii in vector_tiempo:
                ind_aux  = vector_tiempo.index(ii)
                inds_aux = np.where(np.array(x[2]) == ii)
                num_Tx_tiempo[ind_aux] = np.sum(np.array(x[0])[inds_aux])
                vlr_Tx_tiempo[ind_aux] = np.sum(np.array(x[1])[inds_aux])
        return num_Tx_tiempo, vlr_Tx_tiempo

    def caract_pendiente(self,data_producto):
        '''
        Calcula las lineas de tendecias de transacciones y valores a lo largo de
        un rango de tiempo para cada cliente:
        Entrada: 
            data_producto (pandas dataframe): dataframe con historico de todas 
            las transacciones del producto analizado

        Salida:
            ['coef_Tx_tiempo','intercept_Tx_tiempo',
            'coef_Tx_valor','intercept_Tx_valor'] (pandas dataframe) : dataframe
            con pendiente e intercepto de la tendencia de transacciones y 
            pendiente e intercepto de la tendencia de valores.
        '''
        if 'dia' == self.tiempo_pendiente:
            date_aux   = data_producto['tiempo_aux'].apply(
                                        lambda x: dt.strptime(x, '%Y-%m-%d'))
            start_time = np.min(date_aux)
            end_time   = np.max(date_aux)
            vector_tiempo = pd.date_range(start=start_time, 
                                          end=end_time, 
                                          freq='D').astype(str).tolist()

        elif 'mes' == self.tiempo_pendiente:
            date_aux   = data_producto['tiempo_aux'].apply(
                                        lambda x: dt.strptime(x[:7], '%Y-%m'))
            start_time = np.min(date_aux)
            end_time   = np.max(date_aux)
            vector_tiempo = pd.date_range(start=start_time, 
                                          end=end_time, 
                                          freq='MS').astype(str).tolist()

        elif 'semana' == self.tiempo_pendiente:
            date_aux   = data_producto['tiempo_aux'].apply(
                                        lambda x: dt.strptime(x, '%Y-%m-%d'))
            data_producto.insert(data_producto.shape[1],
                                        'tiempo_aux_2',date_aux)
            start_time = np.min(date_aux)
            end_time   = np.max(date_aux)
            vector_tiempo = pd.date_range(start=start_time, 
                                          end=end_time ,
                                          freq='7D').tolist()
            vector_tiempo = [str(ii.year) + '-' + str(ii.week) 
                            for ii in vector_tiempo]
            year_week = data_producto['tiempo_aux_2'].apply(
                                lambda x: str(x.year)+'-'+str(x.week)).values
            data_producto.loc[:,'tiempo_aux'] = year_week.tolist()
            data_producto.drop(columns=['tiempo_aux_2'])

        data_producto_aux = data_producto[['cliente_documento', 
                                           'cant_formularios', 
                                           'vlr_transaccion', 
                                           'tiempo_aux']]

        data_producto_aux = data_producto_aux.groupby(['cliente_documento']
                                                     ).agg(lambda x: list(x))

        data_producto_aux = pd.DataFrame(data_producto_aux.apply(
        lambda x: self.generar_vector_tiempo(vector_tiempo,
                                             x), axis = 1),columns=['Aux_Tx'])

        data_producto_aux['Tx_tiempo'] = data_producto_aux['Aux_Tx'].apply(
                                                                lambda x: x[0]) 
        data_producto_aux['Tx_valor']  = data_producto_aux['Aux_Tx'].apply(
                                                                lambda x: x[1]) 
        data_producto_aux = data_producto_aux.drop(columns=['Aux_Tx'])

        X = np.arange(len(vector_tiempo))
        data_producto_aux['Tx_tiempo'] = data_producto_aux['Tx_tiempo'].apply(
                                                                       np.array)

        aux_regresion_lineal = data_producto_aux['Tx_tiempo'].apply(
                            lambda x: LinearRegression().fit(X[:,np.newaxis],x))

        data_producto_aux['coef_Tx_tiempo']      = aux_regresion_lineal.apply(
                                                         lambda x: x.coef_[0])

        data_producto_aux['intercept_Tx_tiempo'] = aux_regresion_lineal.apply(
                                                       lambda x: x.intercept_)

        data_producto_aux['Tx_valor'] = data_producto_aux['Tx_valor'].apply(
                                                                       np.array)

        aux_regresion_lineal = data_producto_aux['Tx_valor'].apply(
                            lambda x: LinearRegression().fit(X[:,np.newaxis],x))

        data_producto_aux['coef_Tx_valor']      = aux_regresion_lineal.apply(
                                                        lambda x: x.coef_[0])

        data_producto_aux['intercept_Tx_valor'] = aux_regresion_lineal.apply(
                                                      lambda x: x.intercept_)
        return data_producto_aux[['coef_Tx_tiempo',
                                  'intercept_Tx_tiempo',
                                  'coef_Tx_valor',
                                  'intercept_Tx_valor']]

    def caract_clusters(self, tabla_clientes):
        '''Construye dataframe de pertenencia a Cluster'''
        df_clusters_clientes = pd.DataFrame(tabla_clientes['cluster_x'])
        df_clusters_clientes = pd.get_dummies(df_clusters_clientes['cluster_x'],
                                              prefix='cluster')
        return df_clusters_clientes
    
    def caract_rfml(self,data_producto,monto_transacciones,mes_transacciones):
        '''Calcula RFM por producto, por cliente'''
        now = np.max(pd.to_datetime(data_producto['id_tiempo'],
                                    format='%Y%m%d', 
                                    errors = 'coerce'))

        df_aux = data_producto[['cliente_documento','id_tiempo']].copy()
        fechas = pd.to_datetime(df_aux['id_tiempo'],
                                format='%Y%m%d', 
                                errors = 'coerce')

        df_aux.loc[fechas.index,'id_tiempo'] = fechas.values
        df_fecha_transacciones = pd.DataFrame(df_aux.groupby(
                                ['cliente_documento'])['id_tiempo'].apply(list))

        df_fecha_transacciones['ultima_fecha'] = df_fecha_transacciones['id_tiempo'].apply(np.max)

        df_fecha_transacciones['periodo_ultima_compra'] = (
              now - df_fecha_transacciones['ultima_fecha']).astype('<m8[D]')

        RFM_table = pd.concat([monto_transacciones.sum(axis = 1),
                               mes_transacciones.sum(axis = 1),
                               df_fecha_transacciones['periodo_ultima_compra']],
                               axis = 1)

        RFM_table.columns=['monetary_value','frequency','recency']
        quantiles = RFM_table.quantile(q=[0.2,0.4,0.6,0.8])
        quantiles = quantiles.to_dict()

        RFM_Segment = RFM_table.copy()
        RFM_Segment['R_Quartile'] = RFM_Segment['recency'].apply(
                       self.R_Class, args=('recency',quantiles,))

        RFM_Segment['F_Quartile'] = RFM_Segment['frequency'].apply(
                      self.FM_Class, args=('frequency',quantiles,))

        RFM_Segment['M_Quartile'] = RFM_Segment['monetary_value'].apply(
                      self.FM_Class, args=('monetary_value',quantiles,))
        
        RFM_Segment['SIMILARIDAD']=(0.1 * RFM_Segment['R_Quartile'] + 
                                    0.3 * RFM_Segment['F_Quartile'] + 
                                    0.6 * RFM_Segment['M_Quartile'])
        return RFM_Segment

    def caract_agregacion(self, data_producto, agregacion_variable):
        '''
        Calcula caracteristicas cuantificables para la variable indicada teniendo
        en cuenta el nivel de agregación especificado
        '''
        agregacion = agregacion_variable[0]
        variable = agregacion_variable[1]

        if agregacion == 'moda':
            return self.agg_moda(data_producto, variable)

        elif agregacion == 'moda_dummie':
            return self.agg_moda_dummie(data_producto, variable)

        elif agregacion == 'conteo':
            return self.agg_transacciones_pivote(data_producto, variable)

        elif agregacion == 'suma_tx':
            return self.agg_val_trans_suma(data_producto, 'cant_formularios', variable)
        
        elif agregacion == 'suma_vlr':
            return self.agg_val_trans_suma(data_producto, 'vlr_transaccion', variable)

        elif agregacion == 'suma_tx_tiempo':
            return self.agg_val_trans_suma_tiempo(data_producto, 'cant_formularios', variable)

        elif agregacion == 'suma_vlr_tiempo':
            print('entrada a suma_vlt_tiempo')
            return self.agg_val_trans_suma_tiempo(data_producto, 'vlr_transaccion', variable)

    def caracteristicas(self, df_data, tabla_clientes):
        '''Metodo de caracterización de clientes a partir de las variables 
           deseadas establecidas:
           Entrada:
                df_data (pandas dataframe): dataframe con historico de todas las
                transacciones del producto/s analizado/s

                tabla_clientes (pandas dataframe): dataframe con caracteristicas
                de los clientes extraidas del Reto 1

            Salida:
                lista_caracteristicas (pandas dataframe): dataframe con el 
                condensado de caracteristicas calculado

                data_producto (pandas dataframe): dataframe con historico de 
                todas las transacciones del producto/s analizado/s. Se incluyen
                alguna columnas calculadas.
        '''
        data_producto = df_data.loc[df_data['producto_nomb_producto'].isin(
                                                             self.producto)]

        self.date = (data_producto['tiempo_anio'].astype(str) + 
        '-' + data_producto['tiempo_mes'].apply(lambda x: "{:02}".format(x)) + 
        '-' + data_producto['tiempo_dia'].apply(lambda x: "{:02}".format(x)))

        anio_mes = self.date.apply(lambda x: x[:7])

        data_producto.insert(data_producto.shape[1],'tiempo_aux',self.date)
        data_producto.insert(data_producto.shape[1],'tiempo_anio_mes', anio_mes)

        print('\n   Tamaño datos producto : ',data_producto.shape)
        #-----------------------------------------------------------------------
        # Si se establece un perido se elimianan las transacciones antes de 
        # ese periodo de tiempo
        if self.rango_tiempo != None:

            data_producto['tiempo_aux'] = data_producto['tiempo_aux'].apply(
                             lambda x : pd.to_datetime(x,format='%Y-%m-%d'))
            
            ultima_fecha = np.max(data_producto['tiempo_aux'])
            
            if 'anio' == self.rango_tiempo[0]:
                delta_time = ultima_fecha - relativedelta(years = self.rango_tiempo[1])
            elif 'mes' == self.rango_tiempo[0]:
                delta_time = ultima_fecha - relativedelta(months = self.rango_tiempo[1])
            elif 'semana' == self.rango_tiempo[0]:
                delta_time = ultima_fecha - relativedelta(weeks = self.rango_tiempo[1])
            elif 'dia' == self.rango_tiempo[0]:
                delta_time = ultima_fecha - relativedelta(days = self.rango_tiempo[1])
                
            data_producto = data_producto.loc[data_producto['tiempo_aux'] > delta_time]
            
            print('\nTamaño datos producto en rango de tiempo : ', data_producto.shape)    

        # Se eliminan transacciones de clientes perdidos
        #df_tiempo = df_tiempo.loc[index_esporadicos]  
        #index_aux = df_tiempo['periodo_primera_compra'] > self.valor_t
        #df_tiempo = df_tiempo.loc[index_aux,:]

        data_producto['fin_semana'] = data_producto['tiempo_dia_semana'].apply(
             lambda x: 'FinSemana' if x in self.dias_fin_semana else 'EnSemana') 

        mes_transacciones   = self.caract_agregacion(data_producto, 
                                                     self.agregacion_caract['caract_tx_mes'])
        print(mes_transacciones)

        monto_transacciones = self.caract_agregacion(data_producto, 
                                                     self.agregacion_caract['caract_vlr_mes'])

        print(monto_transacciones)
    
        lista_caracteristicas = []

        for caracteristica in self.agregacion_caract.keys():
            
            print('\n Calculando ' + str(caracteristica))

            if caracteristica == 'caract_pendiente':
                lista_caracteristicas.append(self.caract_pendiente(data_producto))
                print('añadida caracteristica pendiente')
            if caracteristica == 'caract_clusters':
                lista_caracteristicas.append(self.caract_clusters(tabla_clientes))
                print('añadida caracteristica clusters')
            if caracteristica == 'caract_rfml':
                lista_caracteristicas.append(self.caract_rfml(data_producto,
                                                            monto_transacciones,
                                                            mes_transacciones))
                print('añadida caracteristica rfml')

            else:
                lista_caracteristicas.append(
                self.caract_agregacion(data_producto,
                                       self.agregacion_caract[caracteristica]))
    
        self.list_features = lista_caracteristicas
        df_caracteristica_clientes = pd.concat(lista_caracteristicas, axis = 1)
        
        return df_caracteristica_clientes, data_producto
   
    def etiquetar_clientes(self, data_producto,df_caracteristica_clientes):
        '''
        Calcula las etiquetas de cliente perdido y habitual, además de descartar
        clientes nuevos y clientes ocasionales:
        Entrada:
            data_producto (pandas dataframe): dataframe con historico de 
                todas las transacciones del producto/s analizado/s.

            df_caracteristica_clientes (pandas dataframe): dataframe con 
            condensado de caracteristicas especificadas

        Salida:
            df_caracteristica_clientes (pandas dataframe): dataframe con 
            condensado de caracteristicas especificadas despues de modificaciones

            df_etiquetas (pandas dataframe): dataframe con etiquetas de cliente
            perdido (0) y cliente habitual (1).
        '''
        # Eliminar clientes esporadicos de df_caracteristicas
        print('DataFrame Caracteristicas Clientes:', 
              df_caracteristica_clientes.shape)

        index_esporadicos  =  (df_caracteristica_clientes.filter(regex ='^transaccion').sum(
                              axis = 1) > self.num_transacciones_minimo)
        df_caracteristica_clientes = df_caracteristica_clientes.loc[index_esporadicos]
        print('DataFrame Caracteristicas Clientes 2:', 
              df_caracteristica_clientes.shape)

        data_producto['tiempo_aux'] = data_producto['tiempo_aux'].apply(
                             lambda x : pd.to_datetime(x,format='%Y-%m-%d'))
        now = ultima_fecha = np.max(data_producto['tiempo_aux'])

        # 1 - Clientes habituales, 0 - Clientes perdidos

        df_tiempo = pd.DataFrame(data_producto.groupby(
            ['cliente_documento'])['tiempo_aux'].apply(list))

        print(df_tiempo['tiempo_aux'].dtypes)

        df_tiempo['ultima_fecha'] = df_tiempo['tiempo_aux'].apply(np.max)
        df_tiempo['primera_fecha'] = df_tiempo['tiempo_aux'].apply(np.min)

        print(df_tiempo['ultima_fecha'].dtypes)
        print(df_tiempo['primera_fecha'].dtypes)
        print(type(now))

        # Dependiendo de los valores de entrada se establence los requerimientos 
        #de ultima y primera transaccion 
        if self.periodo_t == 'dia':
            df_tiempo['periodo_ultima_compra']  = (now - df_tiempo['ultima_fecha']).astype('<m8[D]')
            df_tiempo['periodo_primera_compra'] = (now - df_tiempo['primera_fecha']).astype('<m8[D]')

        if self.periodo_t == 'semana':
            df_tiempo['periodo_ultima_compra']  = (now - df_tiempo['ultima_fecha']).astype('<m8[D]')/7
            df_tiempo['periodo_primera_compra'] = (now - df_tiempo['primera_fecha']).astype('<m8[D]')/7

        if self.periodo_t == 'mes':
            df_tiempo['periodo_ultima_compra']  = (now - 
                         df_tiempo['ultima_fecha']).astype('<m8[D]')/30
            df_tiempo['periodo_primera_compra'] = (now - 
                        df_tiempo['primera_fecha']).astype('<m8[D]')/30

        # Se eliminan transacciones de clientes perdidos
        df_tiempo = df_tiempo.loc[index_esporadicos]  
        index_aux = df_tiempo['periodo_primera_compra'] > self.valor_t
        df_tiempo = df_tiempo.loc[index_aux,:]

        df_caracteristica_clientes = df_caracteristica_clientes.loc[df_tiempo.index]

        print('DataFrame Caracteristicas Clientes 3:', 
              df_caracteristica_clientes.shape)
        
        df_etiquetas = pd.DataFrame(data = np.nan,
                                    index = df_tiempo.index,
                                    columns = ['etiqueta'])
        index_clientes_habituales = df_tiempo.loc[
            (df_tiempo['periodo_ultima_compra'] <= self.valor_t),:].index

        index_clientes_perdidos   = df_tiempo.loc[
            (df_tiempo['periodo_ultima_compra'] > self.valor_t),:].index

        df_etiquetas.loc[index_clientes_habituales,['etiqueta']] = 1 # Clientes habituales
        df_etiquetas.loc[index_clientes_perdidos,['etiqueta']]   = 0 # Clientes Perdidos
        
        return df_caracteristica_clientes, df_etiquetas
 
    def fit(self, df_data, tabla_clientes, *_):
        '''
        Controlador de clase que lanza el proceso de caracterización de los 
        clientes y el proceso de etiquetado.
        Entrada:
            df_data (pandas dataframe): dataframe con historico de 
            todas las transacciones del producto/s analizado/s.

            tabla_clientes (pandas dataframe): dataframe con caracteristicas
            de los clientes extraidas del Reto 1.
        '''
        print('Caracterizando Clientes ...')
        self.df_caracteristica_clientes, data_producto = self.caracteristicas(
                                                      df_data, tabla_clientes)
        self.df_caracteristicas, self.df_etiquetas = self.etiquetar_clientes(
                              data_producto, self.df_caracteristica_clientes)
        print('\nClientes Caracterizados.')
        return self