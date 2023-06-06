from django.shortcuts import render, redirect
from .models import Proyecto
import os
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

import plotly.express as px
from plotly.offline import plot

#from IngeniousPrediction.models import Profesor
#source = "IngeniousPrediction/data/melb_data_uQl4GDv.csv"
# Create your views here.
def index(request):
    return render(request, 'index.html')

#def validar(request):
    #if request.method == 'POST':
        #print(request.POST['email'])
        #profesor = Profesor.object.get(email=request.POST['email'])
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
        # 
        #      
        # 
def showDatos(pk):
    proyecto = Proyecto.objects.get(pk=pk)
    #source = "IngeniousPrediction/data/melb_data_uQl4GDv.csv"
    source = proyecto.data
    context = {}
    context['pk'] = "Grafic"#pk
    #Comienzo del algoritmo
    df = pd.read_csv(source)
    for i in range(df.shape[1]):
        df.columns.values[i] = df.columns.values[i].replace(" ","_")
    
    df2 = df.iloc[np.r_[0:5, -5:0]]
    context['df'] = df2
    
    #Forma del df
    size = df.shape
    context['size'] = size
    
    #Tipos de datos
    tipos = []
    for i in range(df.shape[1]):
        column = df.columns.values[i]
        value = df[column].dtype
        tipos.append(str(column) + ': ' + str(value))
    context['tipos'] = tipos
    
    #Valores nulos
    nulos = []
    for i in range(df.shape[1]):
        column = df.columns.values[i]
        value = df[column].isnull().sum()
        nulos.append(str(column) + ': ' + str(value))
    context['nulos'] = nulos

    #Resumen estadistico de variables numericas
    df3 = df.describe()
    context['df3'] = df3
    
    #Limpieza de datos categoricos
    NuevaMatriz = df.drop(columns=df.select_dtypes('object'))
    ME = NuevaMatriz.iloc[np.r_[0:5, -5:0]]
    context['ME']=ME

    #Correlaciones
    correlaciones = df.corr()
    mask = np.triu(np.ones_like(correlaciones, dtype=np.bool))
    correlaciones = correlaciones.mask(mask)
    #Mapa de calor de correlaciones
    calor = px.imshow(correlaciones, text_auto=True, aspect="auto")

    mapaC = plot({'data': calor}, output_type='div')
    context['corr']=correlaciones
    context['mapaC'] = mapaC

    return(context) 
 
def GUI(request):
    username = 'Eduardo uwu'
    return render(request, 'guiMineria.html',{
        'username' : username,
        # 'data' : Planea
    })

def EDA(request,pk):
    proyecto = Proyecto.objects.get(pk=pk)
    source = proyecto.data
    context = {}
    
    #Comienzo del algoritmo
    df = pd.read_csv(source)
    df2 = df.iloc[np.r_[0:5, -5:0]]
    context['df'] = df2
    
    #Forma del df
    size = df.shape
    context['size'] = size
    
    #Tipos de datos
    tipos = []
    for i in range(df.shape[1]):
        column = df.columns.values[i]
        value = df[column].dtype
        tipos.append(str(column) + ': ' + str(value))
    context['tipos'] = tipos
    
    #Valores nulos
    nulos = []
    for i in range(df.shape[1]):
        column = df.columns.values[i]
        value = df[column].isnull().sum()
        nulos.append(str(column) + ': ' + str(value))
    context['nulos'] = nulos

    #Resumen estadistico de variables numericas
    df3 = df.describe()
    context['df3'] = df3

    #Histogramas
    histogramas = []
    for i in range(df.shape[1]):
        dataType = df.columns.values[i]
        if df[dataType].dtype != object:
            fig = px.histogram(df, x=df.columns[i])
            
            plot_div = plot({'data': fig}, output_type='div')
            histogramas.append(plot_div)

    context['plot_div'] = histogramas

    #Diagramas de caja
    cajas = []
    for i in range(df.shape[1]):
        dataType = df.columns.values[i]
        if df[dataType].dtype != object:
            fig = px.box(df, x=df.columns[i])            
            plot_div = plot({'data': fig}, output_type='div')
            cajas.append(plot_div)
    context['diagramsCaja'] = cajas

    #Verificar que el dataframe contenga variables no numericas
    try:
        df.describe(include='object')
    except:
        objects = False
    else:
        objects = True
    context['flag'] = objects

    #Toma de decision en caso de haber variables no numericas
    if(objects == True):
        #Distribucion variables categoricas
        df4 = df.describe(include='object')
        context['df4']=df4
        #Plots de las distribuciones
        Cat = []
        for col in df.select_dtypes(include='object'):
            if df[col].nunique()< 10:
                fig = px.histogram(df, y=col)
                plot_div = plot({'data': fig}, output_type='div')
                Cat.append(plot_div)

        context['Cat']=Cat

        #Agrupacion por variables categoricas
        groups = []
        for col in df.select_dtypes(include='object'):
            if df[col].nunique() < 10:
                dataG = df.groupby(col).agg(['mean'])
                print(dataG)
                groups.append(dataG)
        context['groups']=groups
    
    #Correlaciones
    correlaciones = df.corr()
    mask = np.triu(np.ones_like(correlaciones, dtype=np.bool))
    correlaciones = correlaciones.mask(mask)
    #Mapa de calor de correlaciones
    calor = px.imshow(correlaciones, text_auto=True, aspect="auto")
    mapaC = plot({'data': calor}, output_type='div')
    context['mapaC'] = mapaC


    return render(request, 'EDA.html', context)

def PCA(request):
    return render(request, 'PCA.html')

def ArbolDecision(request):
    return render(request, 'ArbolDecision.html')

def BosqueAleatorio(request):
    return render(request, 'BosqueAleatorio.html')

def KMeans(request):
    return render(request, 'K-Means.html')
    

def lista_Proyectos(request):
    proyectos = Proyecto.objects.all()
    return render(request, 'Proyectos.html', {
        'proyectos': proyectos
    })

def crea_Proyecto(request):
    Nombre = request.POST['Nombre']
    Desc = request.POST['descripcion']
    URL = 'xd'
    data = request.FILES['data']
    
    if Desc == '':
        Desc = "Descripcion no proporcionada."

    ext = os.path.splitext(data.name)[1]
    print(ext)
    valid_extensions = '.csv'
    if not ext.lower() in valid_extensions:
        return redirect('/ErrorProyecto')
    else:
        Proyecto.objects.create(Nombre=Nombre, descripcion=Desc, URL = URL, data = data)
        return redirect('project_list')

def delete_project(request, pk):
    proyecto = Proyecto.objects.get(pk=pk)
    proyecto.delete()
    return redirect('project_list')

def ErrorProyecto(request):
    proyectos = Proyecto.objects.all()
    return render(request, 'ProyectosError.html', {
        'proyectos': proyectos
    })
    
