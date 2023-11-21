import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple

def grafico_frecuencia_acumulada(data: pd.DataFrame, categorical:str, target:str, title:str, colors: List[str], x_axis_rotation:int= 0):
    """Función para graficar la frecuencia acumulada de una variable categórica con respecto a una variable objetivo.

    Args:
        data (pd.DataFrame): Dataframe con los datos a graficar.
        categorical (str): Variable categórica.
        target (str): Variable objetivo.
        title (str): Título del gráfico.
        colors (List[str]): Lista de colores para el gráfico.
        x_axis_rotation (int, optional): Rotación del eje x. Defaults to 0.
    """
    t1 = pd.crosstab(data[categorical], data[target]) 
    p1 = t1.div(t1.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(10, 7))
    p1.plot(kind='bar', stacked=True, ax=ax, alpha = 0.7, color = colors)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('{}'.format(categorical), fontsize=12)
    plt.xticks(rotation=x_axis_rotation)
    ax.set_ylabel('Frecuencia', fontsize=12)
    plt.legend(title=f'{target}')
    plt.show()