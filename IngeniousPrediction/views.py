from django.shortcuts import render
# from .models import Profesor

#Dependencias analisis
#%matplotlib inline       
import pandas as pd                     # Para la manipulación y análisis de datos
import numpy as np                      # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt         # Para la generación de gráficas a partir de los datos
import seaborn as sns                   # Para la visualización de datos basado en matplotlib         
# Para generar imágenes dentro del cuaderno
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler  
from django.contrib import messages

from IngeniousPrediction.models import Profesor

# Create your views here.
def index(request):
    return render(request, 'index.html')

def validar(request):
    if request.method == 'POST':
        print(request.POST['email'])
        profesor = Profesor.object.get(email=request.POST['email'])
        # try:
        #     profesor = Profesor.object.get(email=request.POST['email'])
        #     print(profesor)
        #     messages.success(request, "Bienvenido")
        #     return render(request, 'EDA.html')
        # except:
        #     messages.error(request, "Usuario No Encontrado")
        #     print("fallo")
        #     return render(request, 'index.html')
        # username= request.POST["email"]
        # password = request.POST["psw"]
        

def GUI(request):
    # SourceDocument = pd.read_csv("IngeniousPrediction/data/PlaneaDataSet.csv")
    # Planea = SourceDocument

    #Paso 1: Descripción de la estructura de los datos
    # 1) Forma del DataFrame
    # 
    # EstDatos = Planea.shape
    # # 2) Tipos de datos
    # TypeDatos = Planea.dtypes
    # #Paso 2: Identificacion de datos faltantes
    # NullDatos = Planea.isnull().sum()
    # NotNullDatos = Planea.info()

    #query = Profesor.objects.get(profesor_id=1)
    username = 'Eduardo uwu'
    return render(request, 'guiMineria.html',{
        'username' : username,
        # 'data' : Planea
    })

def EDA(request):
    return render(request, 'EDA.html')

def PCA(request):
    return render(request, 'PCA.html')

def ArbolDecision(request):
    return render(request, 'ArbolDecision.html')

def BosqueAleatorio(request):
    return render(request, 'BosqueAleatorio.html')

def KMeans(request):
    return render(request, 'K-Means.html')
    