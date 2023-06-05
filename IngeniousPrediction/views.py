from django.shortcuts import render
from .models import Profesor

#Dependencias analisis
#%matplotlib inline       
import pandas as pd                     # Para la manipulación y análisis de datos
import numpy as np                      # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt         # Para la generación de gráficas a partir de los datos
import seaborn as sns                   # Para la visualización de datos basado en matplotlib         
# Para generar imágenes dentro del cuaderno
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler  


# Create your views here.
def index(request):
    return render(request, 'index.html')

def GUI(request):
    SourceDocument = pd.read_csv("IngeniousPrediction/data/PlaneaDataSet.csv")
    Planea = SourceDocument

    #Paso 1: Descripción de la estructura de los datos
    # 1) Forma del DataFrame
    # 
    EstDatos = Planea.shape
    # 2) Tipos de datos
    TypeDatos = Planea.dtypes
    #Paso 2: Identificacion de datos faltantes
    NullDatos = Planea.isnull().sum()
    NotNullDatos = Planea.info()

    #query = Profesor.objects.get(profesor_id=1)
    username = 'Eduardo uwu'
    return render(request, 'guiMineria.html',{
        'username' : username,
        'data' : Planea
    })