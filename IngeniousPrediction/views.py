import email
from django.shortcuts import render, redirect
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
        try:
            profesor = Profesor.objects.get(email=request.POST['email'])
            print(profesor)
            messages.success(request, "Bienvenido")
            return redirect('EDA')
        except:
            messages.error(request, "Usuario No Encontrado")
            return render(request,'index.html')
        username= request.POST["email"]
        password = request.POST["psw"]
def registrar(request):
    if request.method == 'POST':
        p_nombre = request.POST['nombre']
        p_apellido = request.POST['apellidos']
        p_correo = request.POST['correo']
        p_password1 = request.POST['psw']
        p_password2 = request.POST['pswCon'] 
        if p_password1 == p_password2:
            profe = Profesor(nombre=p_nombre, apellido=p_apellido, email=p_correo, passwd=p_password1)
            profe.save()
            print("usuario creado")
            return redirect('EDA')
        else:
            messages.error(request, "Contraseña no igual")
            return render(request,'index.html')

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
    