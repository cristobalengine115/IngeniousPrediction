from django.shortcuts import render
from .models import Proyecto

#Dependencias analisis
#%matplotlib inline       
import pandas as pd                     # Para la manipulación y análisis de datos
import numpy as np                      # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt         # Para la generación de gráficas a partir de los datos
import plotly.express as px
from plotly.offline import plot
import seaborn as sns                   # Para la visualización de datos basado en matplotlib         
# Para generar imágenes dentro del cuaderno
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler  


# Create your views here.
def index(request):
    return render(request, 'index.html')


################ Primeros Pasos algoritmos ##################
def showDatos(pk):
    project = Proyecto.objects.get(pk=pk)
    source = project.data
    context = {}
    context['pk'] = pk
    


    datasetsource = pd.read_csv(source)
    for i in range(datasetsource.shape[1]):
        datasetsource.columns.values[i] = datasetsource.columns.values[i].replace(" ","_")
    
    datasetsourcetemp = datasetsource.iloc[np.r_[0:5, -5:0]]
    context['datasetsource'] = datasetsourcetemp
    
    ############## Forma Dataset ##############
    size = datasetsource.shape
    context['size'] = size
    
    ########### Determinar tipo de datos ###############
    datatypes = []
    for i in range(datasetsource.shape[1]):
        column = datasetsource.columns.values[i]
        value = datasetsource[column].dtype
        datatypes.append(str(column) + ': ' + str(value))
    context['datatypes'] = datatypes
    
    ############### Determinar valores nulos #############
    nulldata = []
    for i in range(datasetsource.shape[1]):
        column = datasetsource.columns.values[i]
        value = datasetsource[column].isnull().sum()
        nulldata.append(str(column) + ': ' + str(value))
    context['nulldata'] = nulldata

    ################ Resumen Estadistico ################
    datasetsourcehlp = datasetsource.describe()
    context['datasetsourcehlp'] = datasetsourcehlp
    
    #Limpieza de datos categoricos
    NuevaMatriz = df.drop(columns=df.select_dtypes('object'))
    ME = NuevaMatriz.iloc[np.r_[0:5, -5:0]]
    context['ME']=ME

    ############## Correlaciones ################
    corrdata = datasetsource.corr()
    mask = np.triu(np.ones_like(corrdata, dtype=np.bool))
    corrdata = corrdata.mask(mask)

    ########### Mapa de calor de correlaciones ###########
    heat = px.imshow(corrdata, text_auto=True, aspect="auto")

    heatmap = plot({'data': heat}, output_type='div')
    context['corr'] = corrdata
    context['heatmap'] = heatmap

    return(context) 


####################### Algoritmo EDA ######################
def AlgoritmoEDA(request):
    #proyecto = Proyecto.objects.get(pk=pk)
    #source = proyecto.data
    source = "IngeniousPrediction/data/PlaneaDataSet.csv"

    context = {}
    
    ########### Lectura de datos #############
    datasetsource = pd.read_csv(source)
    datasetsourcetemp = datasetsource.iloc[np.r_[0:5, -5:0]]
    context['datasetsource'] = datasetsourcetemp
    
    ############ Adquirir forma de los datos ##########
    size = datasetsource.shape
    context['size'] = size
    
    ############# Determinar el tipo de datos ############
    datatypes = []
    for i in range(datasetsource.shape[1]):
        column = datasetsource.columns.values[i]
        value = datasetsource[column].dtype
        datatypes.append(str(column) + ': ' + str(value))
    context['datatypes'] = datatypes
    
    ############ Determinar los valores nulos ############
    nulldata = []
    for i in range(datasetsource.shape[1]):
        column = datasetsource.columns.values[i]
        value = datasetsource[column].isnull().sum()
        nulldata.append(str(column) + ': ' + str(value))
    context['nulldata'] = nulldata

    ############# Resumen estadistico de variables numericas ##############
    datasetsourcehlp = datasetsource.describe()
    context['datasetsourcehlp'] = datasetsourcehlp

    #Histogramas
    histograms = []
    for i in range(datasetsource.shape[1]):
        typedata = datasetsource.columns.values[i]
        if datasetsource[typedata].dtype != object:
            fig = px.histogram(datasetsource, x=datasetsource.columns[i])
            
            plot_div = plot({'data': fig}, output_type='div')
            histograms.append(plot_div)

    context['histograms'] = histograms

    ########## Gatito Box ##########
    kittybox = []
    for i in range(datasetsource.shape[1]):
        dataType = datasetsource.columns.values[i]
        if datasetsource[dataType].dtype != object:
            fig = px.box(datasetsource, x=datasetsource.columns[i])            
            plot_div = plot({'data': fig}, output_type='div')
            kittybox.append(plot_div)
    context['kittybox'] = kittybox

    #Verificar que el dataframe contenga variables no numericas
    try:
        datasetsource.describe(include='object')
    except:
        objects = False
    else:
        objects = True
    context['flag'] = objects

    ######### Toma de decision en caso de haber variables no numericas ##########
    if(objects == True):
        #Distribucion variables categoricas
        datasetsourcecat = datasetsource.describe(include='object')
        context['datasetsourcecat'] = datasetsourcecat
        ######## Plots de las distribuciones #######
        Cat = []
        for col in datasetsource.select_dtypes(include='object'):
            if datasetsource[col].nunique()< 10:
                fig = px.histogram(datasetsource, y=col)
                plot_div = plot({'data': fig}, output_type='div')
                Cat.append(plot_div)

        context['Cat']=Cat

        #Agrupacion por variables categoricas
        groups = []
        for col in datasetsource.select_dtypes(include='object'):
            if datasetsource[col].nunique() < 10:
                dataG = datasetsource.groupby(col).agg(['mean'])
                print(dataG)
                groups.append(dataG)
        context['groups']=groups
    
    ############ Correlaciones########
    corrdata = datasetsource.corr()
    mask = np.triu(np.ones_like(corrdata, dtype=np.bool))
    corrdata = corrdata.mask(mask)
    ####### HeatMap ###########
    heat = px.imshow(corrdata, text_auto=True, aspect="auto")
    heatmap = plot({'data': heat}, output_type='div')
    context['heatmap'] = heatmap
    return render(request, 'AlgoritmoEDA.html', context)



def GUI(request):
 
    #query = Profesor.objects.get(profesor_id=1)
    username = 'Eduardo uwu'
    return render(request, 'guiMineria.html',{
        'username' : username,
        'eda': AlgoritmoEDA
        #'data' : Planea
    })