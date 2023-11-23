import os
import pandas as pd
from typing import Dict, List, Tuple

def read_data(month_name:str)-> Dict[str, pd.DataFrame]:
    
    db: Dict[str, pd.DataFrame] = {}
    filepath = os.path.dirname(__file__)
    folderpath = os.path.join(filepath, '..', 'data', 'data', month_name, 'CSV')
    files = ['Ocupados.CSV','Características generales, seguridad social en salud y educación.CSV','Datos del hogar y la vivienda.CSV']
    names = ['ocupados', 'caracteristicas', 'hogar']
    for name, file in zip(names, files):
        db[name] = pd.read_csv(os.path.join(folderpath, file), sep=';', encoding='latin-1',low_memory=False)
    
    return db


def add_occupation_status(df, ocupados):
    merged_df = pd.merge(df, ocupados, on=['DIRECTORIO', 'SECUENCIA_P'], how='left')
    df = df.copy()
    df['Occupation_Status'] = ~merged_df['P6040'].isnull()
    return df

def process_data(month_name:str)-> pd.DataFrame:
    
    db = read_data(month_name)
    
    # Hogar
    hogar_cols = ['DIRECTORIO','SECUENCIA_P','PERIODO','HOGAR','P4000', 'P4030S1','P4030S1A1','P4030S2','P4030S3','P4030S4',
                  'P4030S5','P5222S1','P5222S2','P5222S3','P5222S4','P5222S5','P5222S6','P5222S7','P5222S8','P5222S8A1',
                  'P5222S9','P5222S10','P6008','DPTO','AREA','CLASE']
    hogares = db['hogar'][hogar_cols]
    
    # Características Personales
    persona_cols = ['DIRECTORIO','SECUENCIA_P','ORDEN','PT','POB_MAY18','P3271','P6040','P6050','P6083','P6081','P2057','P2059',
                     'P2061','P6080','P6070','P6160','P6170','P3041','P3042','P3042S2','P3043','P3043S1','P3039']
    personas = db['caracteristicas'][persona_cols]
    
    
    # Merge data
    result_personas = pd.merge(personas, hogares, on=['DIRECTORIO','SECUENCIA_P'], how='left')
    
    # Filtrar por jovenes
    result_jovenes = result_personas[result_personas['P6040'].isin(range(18, 29))]
    
    # Ocupados
    result_jovenes = add_occupation_status(result_jovenes, db['ocupados'])
    result_jovenes['Occupation_Status'] = result_jovenes['Occupation_Status'].isnull().astype(int)
    
    return result_jovenes

    