"""Modulo con la funciones que se utilizan para los calculos estadísticos de un dataframe
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import chi2_contingency

def calculate_woe_iv(dataframe : pd.DataFrame, categorical_var:str, target_var:str)-> Tuple[Dict[str,float], Dict[str,float]]:
    """Función que calcula el WOE y IV para una variable categórica respecto a una variable objetivo

    Args:
        dataframe (pd.DataFrame): Dataframe con los datos
        categorical_var (str): Variable sobre la que se quiere calcular el WOE y IV
        target_var (str): Variable objetivo

    Returns:
        Tuple[Dict[str,float], Dict[str,float]]: Diccionarios con el WOE y IV para cada categoría de la variable categórica
    """
    
    # Cálculo de la tabla de frecuencia para la variable categórica respecto a la variable objetivo
    cruce = pd.crosstab(dataframe[categorical_var], dataframe[target_var])
    
    # Cálculo del total de positivos y negativos
    total_posivitos = cruce[1].sum()
    total_negativos = cruce[0].sum()

    # Definición de los diccionarios de WOE y IV
    WOE: Dict[str, float] = {}
    IV : Dict[str, float] = {}
    
    # Cálculo de la proporción de positivos y negativos para cada categoría de la variable
    for categoria in cruce.index:
        eventos_positivos = cruce.loc[categoria, 1]
        eventos_negativos = cruce.loc[categoria, 0]

        # Cálculo del WOE y IV para cada categoría, en el caso de que no haya positivos o negativos se asigna un valor de 0.0001
        ratio_positivos = (eventos_positivos / total_posivitos) if eventos_positivos != 0 else 0.0001
        ratio_negativos = (eventos_negativos / total_negativos) if eventos_negativos != 0 else 0.0001

        # Calculo del WOE y IV
        woe_categoria = np.log(ratio_positivos / ratio_negativos)
        information_value = (ratio_positivos - ratio_negativos) * woe_categoria

        # Se asigna el WOE y IV a los diccionarios
        WOE[categoria] = woe_categoria
        IV[categoria] = information_value
    
    return WOE, IV


# Función para realizar la prueba CHI2 de normalidad
def chi2_test_of_normality(dataframe: pd.DataFrame, var1: str, var2: str) -> float:
    """Función que realiza la prueba CHI2 de normalidad para dos variables categóricas

    Args:
        dataframe (pd.DataFrame): Dataframe con los datos
        var1 (str): Variable categórica 1
        var2 (str): Variable categórica 2

    Returns:
        float: P-valor de la prueba CHI2 de normalidad
    """
    
    # Create a contingency table for the two categorical variables
    contingency_table = pd.crosstab(dataframe[var1], dataframe[var2])

    # Perform the chi-square test of normality
    chi2, p_value, _, _ = chi2_contingency(contingency_table)

    return p_value
