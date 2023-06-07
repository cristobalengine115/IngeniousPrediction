from django.shortcuts import render, redirect

from .models import Profesor, Proyecto
import os
#Dependencias analisis
#%matplotlib inline       
import pandas as pd                     # Para la manipulación y análisis de datos
import numpy as np                      # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt         # Para la generación de gráficas a partir de los datos
import plotly.express as px
from plotly.offline import plot
import seaborn as sns                   # Para la visualización de datos basado en matplotlib         

from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report,mean_squared_error, mean_absolute_error, r2_score

from django.contrib import messages
import plotly.express as px
from plotly.offline import plot

BASE_DIR = Path(__file__).resolve().parent.parent

# Create your views here.
def index(request):
    return render(request, 'index.html')

def validar(request):
    if request.method == 'POST':
        try:
            profesor = Profesor.objects.get(email=request.POST['email'])
            print(profesor)
            messages.success(request, "Bienvenido")
            return redirect('inicioGUI')
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
            return redirect('inicioGUI')
        else:
            messages.error(request, "Contraseña no igual")
            return render(request,'index.html')

def registrarProyecto(request):
    if request.method == 'POST':
        p_nombre = request.POST['project_Name']
        p_url = request.POST['project_URL']
        p_descripcion = request.POST['project_Desc']
        p_data = request.POST['project_Data']
        proyecto = Proyecto(nombre=p_nombre, descripcion=p_descripcion, URL=p_url, data=p_data)
        proyecto.save()
        print("usuario creado")
        return redirect('inicioGUI')


def inicioGUI(request):
    return render(request, 'inicioGUI.html')

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

def PCA(request,pk):
    proyecto = Proyecto.objects.get(pk=pk)
    source = proyecto.data
    context = {}
    context['pk'] = pk
    #Comienzo del algoritmo
    df = pd.read_csv(source)
    for i in range(df.shape[1]):
        df.columns.values[i] = df.columns.values[i].replace(" ","_")
    df2 = df.iloc[np.r_[0:5, -5:0]]
    context['df']=df2
    
    #Forma del df
    size = df.shape
    context['size']=size
    
    #Correlaciones
    correlaciones = df.corr()
    mask = np.triu(np.ones_like(correlaciones, dtype=np.bool))
    correlaciones = correlaciones.mask(mask)
    #Mapa de calor de correlaciones
    calor = px.imshow(correlaciones, text_auto=True, aspect="auto")

    mapaC = plot({'data': calor}, output_type='div')
    context['corr']=correlaciones
    context['mapaC'] = mapaC

    #Estandarizacion de datos
    Estandarizar = StandardScaler()
    NuevaMatriz = df.drop(columns=df.select_dtypes('object'))
    NuevaMat = NuevaMatriz.dropna()
    MEstandarizada = Estandarizar.fit_transform(NuevaMat)     
    ME = pd.DataFrame(MEstandarizada, columns=NuevaMat.columns)
    ME2 = ME.iloc[np.r_[0:5, -5:0]]
    context['ME']=ME2

    #Instancia de componente PCA
    PCAOut = []
    pca = PCA(n_components=None)
    pca.fit(MEstandarizada)     
    pcaPrint = pca.components_
    for data in pcaPrint:
        PCAOut.append(data)
    context['pca1'] = PCAOut

    #Numero componentes
    Varianza = pca.explained_variance_ratio_
    context['Var']=Varianza
    nComp = 1
    VarAc = 0
    while True:
        VarAc = sum(Varianza[0:nComp])
        if VarAc < 0.9:
            nComp +=1
            print(nComp)
        else:
            if VarAc > 0.9:
                VarAc= sum(Varianza[0:nComp-1])
                nComp -=1
            break
    
    context['nComp']=nComp
    context['VarAc']=VarAc

    #Grafica Varianza acumulada
    figV = px.line(np.cumsum(pca.explained_variance_ratio_), markers=True)
    figV.update_xaxes(title_text='Numero de componentes')
    figV.update_yaxes(title_text='Varianza acumulada')
    figVar = plot({'data': figV}, output_type='div')

    context['figVar']=figVar

    #Paso 6
    CargasComponentes = pd.DataFrame(abs(pca.components_[0:nComp]), columns=NuevaMat.columns)
    context['CargasC']=CargasComponentes

    return render(request, 'PCA.html',context)

def PCA2(request, pk):
    proyecto = Proyecto.objects.get(pk=pk)
    colDrop = request.POST.getlist('columnas')
    source = proyecto.data
    context = {}
    context['pk'] = pk
    df = pd.read_csv(source)
    nDf = df.drop(columns=colDrop)
    outDF = nDf.iloc[np.r_[0:5, -5:0]]
    context['ndf']=outDF
    #Forma del nuevo df
    size = nDf.shape
    context['size']=size
    return render(request, 'PCA2.html', context)

def ArbolDecision(request,pk,algType):
    context = showDatos(pk)
    flag = bool
    context['type'] = algType

    match algType:
        case 'P':
            context['AlgName'] = 'Pronostico'
            flag = True
        case 'C':
            context['AlgName'] = 'Clasificacion'
            flag = False
    context['flag'] = flag
    return render(request, 'ArbolDecision.html')

def ArbolDecisionext(request, pk, algType):
    proyecto = Proyecto.objects.get(pk=pk)
    context = {}
    context['pk'] = pk
    context['type'] = algType
    flag = bool
    #Obtencion de Var Cat y Pred
    predictoras = request.POST.getlist('predictora')
    pronosticar = request.POST['pronostico']
    
    #verificar que no hayas elegido la misma columna en ambos
    for i in range (len(predictoras)):
        if(predictoras[i] == pronosticar):
            return redirect('/ADError/{}/{}'.format(pk,algType))
    
    if( len(predictoras) <= 0):
        return redirect('/ADError/{}/{}'.format(pk,algType))
    #Paso usual
    source = proyecto.data
    df = pd.read_csv(source) 
    for i in range(df.shape[1]):
        df.columns.values[i] = df.columns.values[i].replace(" ","_")   
    
    #Limpiamos de nuevo
    NuevaMat = df.dropna() 


    #Seleccion variables Predictoras y de pronostico
    aux = pd.DataFrame(NuevaMat[predictoras])
    X = np.array(NuevaMat[predictoras])
    Xout = pd.DataFrame(data=X, columns=aux.columns.values)
    Xout.to_csv(os.path.join(BASE_DIR, 'IngeniousPrediction/data/info/X.csv'), index=False)
    context['X'] = Xout.iloc[np.r_[0:5, -5:0]]

    Y = np.array(NuevaMat[pronosticar])
    Yout = pd.DataFrame(Y)
    
    Yout.to_csv(os.path.join(BASE_DIR, 'IngeniousPrediction/data/info/Y.csv'), index=False)
    context['Y'] = Yout.iloc[np.r_[0:5, -5:0]]

    #Division de los datos
    X_train, X_dos, Y_train, Y_dos = model_selection.train_test_split(X, Y, 
                                                                                    test_size = 0.2, 
                                                                                    random_state = 0,
                                                                                    shuffle = True)
    match algType:
        case 'P':
            context['AlgName'] = 'Pronostico'
            #Entrenamiento
            ModeloAD = DecisionTreeRegressor(random_state=0)
            
            #Visualizacion de datos de prueba
            Xtest = pd.DataFrame(X_dos)
            context['Xtest'] = Xtest.iloc[np.r_[0:5, -5:0]]
            
            flag = True
            
        case 'C':
            context['AlgName'] = 'Clasificacion'
            #Entrenamiento
            ModeloAD = DecisionTreeClassifier(random_state=0)
            flag = False

    context['flag'] = flag
    
    ModeloAD.fit(X_train, Y_train)
    
    #Se genera el pronóstico
    Y_Pronostico = ModeloAD.predict(X_dos)
    Ypronostico = pd.DataFrame(Y_Pronostico)
    context['YPron'] = Ypronostico.iloc[np.r_[0:5, -5:0]]

    #Comparacion entre pronostico y prueba
    Valores = pd.DataFrame(Y_dos, Y_Pronostico)
    Valores2 = Valores.reset_index()
    ValoresOut = Valores2.rename(columns={Valores2.columns[0]: 'Prueba', Valores2.columns[1]: 'Pronostico'})
    context['Valores'] = ValoresOut.iloc[np.r_[0:5, -5:0]]

    match algType:
        case 'P':

            #Obtencion del ajuste de Bondad
            Score = r2_score(Y_dos, Y_Pronostico)
            context['Score'] = Score

            #Criterios
            criterios = []
            criterios.append(ModeloAD.criterion)
            criterios.append(mean_absolute_error(Y_dos, Y_Pronostico))
            criterios.append(mean_squared_error(Y_dos, Y_Pronostico))
            criterios.append(mean_squared_error(Y_dos, Y_Pronostico, squared=False))
            context['criterios'] = criterios

        case 'C':

            #Obtencion del ajuste de Bondad
            Score = accuracy_score(Y_dos, Y_Pronostico)
            context['Score'] = Score

            #Matriz de clasificacion
            ModeloClasificacion1 = ModeloAD.predict(X_dos)
            Matriz_Clasificacion1 = pd.crosstab(Y_dos.ravel(), 
                                        ModeloClasificacion1, 
                                        rownames=['Actual'], 
                                        colnames=['Clasificación']) 
            context['MClas']=Matriz_Clasificacion1

            #Criterios
            criterios = []
            criterios.append(ModeloAD.criterion)
            criterios.append(accuracy_score(Y_dos, Y_Pronostico))
            context['criterios'] = criterios
            ReporteC =classification_report(Y_dos, Y_Pronostico, output_dict=True)
            RepClas = pd.DataFrame(ReporteC).transpose()
            context['ReporteClas'] = RepClas         

    #Dataframe de la importancia de variables
    Importancia = pd.DataFrame({'Variable': list(NuevaMat[predictoras]),
                            'Importancia': ModeloAD.feature_importances_}).sort_values('Importancia', ascending=False)
    context['Imp'] = Importancia

    #Reporte o arbol en texto

    Reporte = export_text(ModeloAD, feature_names = predictoras)
    RepOut = []
    RepOut = Reporte.split("\n")
    context['reportes'] = RepOut

    #Eleccion para nuevo pronostico
    predictorasOut = NuevaMat[predictoras]
    context['Pred'] = predictorasOut.iloc[np.r_[0:5, -5:0]]

    return render(request, 'ArbolDecision2.html', context)

def ArbolDecisionext2(request,pk, algType):
    context = {}
    context['pk'] = pk
    context['type'] = algType
    Val = request.POST.getlist('Nvals')
    X = pd.read_csv(os.path.join(BASE_DIR, 'IngeniousPrediction/data/info/X.csv'))
    Y = pd.read_csv(os.path.join(BASE_DIR, 'IngeniousPrediction/data/info/Y.csv'))

    #Division de los datos
    X_train, X_dos, Y_train, Y_dos = model_selection.train_test_split(X, Y, 
                                                                        test_size = 0.2, 
                                                                        random_state = 0, 
                                                                        shuffle = True)
    
    match algType:
        case 'P':
            context['AlgName'] = 'Pronostico'
            #Entrenamiento
            ModeloAD = DecisionTreeRegressor(random_state=0)            
            flag = True
            
        case 'C':
            context['AlgName'] = 'Clasificacion'
            #Entrenamiento
            ModeloAD = DecisionTreeClassifier(random_state=0)
            flag = False

    context['flag'] = flag
    
    ModeloAD.fit(X_train, Y_train)

    col = list(X.columns)
    datoOut = {}
    for i in range(X.shape[1]):
        datoOut[col[i]] = float(Val[i])
    Npron = pd.DataFrame(datoOut, index=[0])
    context['DfN'] = Npron
    resultado = ModeloAD.predict(Npron)
    context['resultado'] = resultado

    return render(request, 'Arboles/ArbolDecision3.html', context)

def BosqueAleatorio(request,pk,algType):
    context = showDatos(pk)
    flag = bool
    context['type'] = algType

    match algType:
        case 'P':
            context['AlgName'] = 'Pronostico'
            flag = True
        case 'C':
            context['AlgName'] = 'Clasificacion'
            flag = False
    context['flag'] = flag
    return render(request, 'BosqueAleatorio.html', context)

def BosqueAleatorioext(request, pk, algType):
    proyecto = Proyecto.objects.get(pk=pk)
    context = {}
    context['pk'] = pk
    context['type'] = algType
    flag = bool
    #Obtencion de Var Cat y Pred
    predictoras = request.POST.getlist('predictora')
    pronosticar = request.POST['pronostico']
    
    #verificar que no hayas elegido la misma columna en ambos y que se haya escogido al menos una variable predictora
    for i in range (len(predictoras)):
        if(predictoras[i] == pronosticar):
            return redirect('/BAError/{}/{}'.format(pk,algType))
    
    if( len(predictoras) <= 0):
        return redirect('/BAError/{}/{}'.format(pk,algType))

    #Paso usual
    source = proyecto.data
    df = pd.read_csv(source)
    for i in range(df.shape[1]):
        df.columns.values[i] = df.columns.values[i].replace(" ","_")
    
    #Limpiamos de nuevo
    NuevaMat = df.dropna() 


    #Seleccion variables Predictoras y de pronostico
    aux = pd.DataFrame(NuevaMat[predictoras])
    X = np.array(NuevaMat[predictoras])
    Xout = pd.DataFrame(data=X, columns=aux.columns.values)
    Xout.to_csv(os.path.join(BASE_DIR, 'IngeniousPrediction/data/info/X.csv'), index=False)
    context['X'] = Xout.iloc[np.r_[0:5, -5:0]]

    Y = np.array(NuevaMat[pronosticar])
    Yout = pd.DataFrame(Y)
    
    Yout.to_csv(os.path.join(BASE_DIR, 'IngeniousPrediction/data/info/Y.csv'), index=False)
    context['Y'] = Yout.iloc[np.r_[0:5, -5:0]]

    #Division de los datos
    X_train, X_dos, Y_train, Y_dos = model_selection.train_test_split(X, Y, 
                                                                                    test_size = 0.2, 
                                                                                    random_state = 0,
                                                                                    shuffle = True)
    match algType:
        case 'P':
            context['AlgName'] = 'Pronostico'
            #Entrenamiento
            ModeloBA = RandomForestRegressor(random_state=0)
            
            #Visualizacion de datos de prueba
            Xtest = pd.DataFrame(X_dos)
            context['Xtest'] = Xtest.iloc[np.r_[0:5, -5:0]]
            
            flag = True
            
        case 'C':
            context['AlgName'] = 'Clasificacion'
            #Entrenamiento
            ModeloBA = RandomForestClassifier(random_state=0)
            flag = False

    context['flag'] = flag
    
    ModeloBA.fit(X_train, Y_train)
    
    #Se genera el pronóstico
    Y_Pronostico = ModeloBA.predict(X_dos)
    Ypronostico = pd.DataFrame(Y_Pronostico)
    context['YPron'] = Ypronostico.iloc[np.r_[0:5, -5:0]]

    #Comparacion entre pronostico y prueba
    Valores = pd.DataFrame(Y_dos, Y_Pronostico)
    Valores2 = Valores.reset_index()
    ValoresOut = Valores2.rename(columns={Valores2.columns[0]: 'Prueba', Valores2.columns[1]: 'Pronostico'})
    context['Valores'] = ValoresOut.iloc[np.r_[0:5, -5:0]]

    match algType:
        case 'P':

            #Obtencion del ajuste de Bondad
            Score = r2_score(Y_dos, Y_Pronostico)
            context['Score'] = Score

            #Criterios
            criterios = []
            criterios.append(ModeloBA.criterion)
            criterios.append(mean_absolute_error(Y_dos, Y_Pronostico))
            criterios.append(mean_squared_error(Y_dos, Y_Pronostico))
            criterios.append(mean_squared_error(Y_dos, Y_Pronostico, squared=False))
            context['criterios'] = criterios

        case 'C':

            #Obtencion del ajuste de Bondad
            Score = accuracy_score(Y_dos, Y_Pronostico)
            context['Score'] = Score

            #Matriz de clasificacion
            ModeloClasificacion1 = ModeloBA.predict(X_dos)
            Matriz_Clasificacion1 = pd.crosstab(Y_dos.ravel(), 
                                        ModeloClasificacion1, 
                                        rownames=['Actual'], 
                                        colnames=['Clasificación']) 
            context['MClas']=Matriz_Clasificacion1

            #Criterios
            criterios = []
            criterios.append(ModeloBA.criterion)
            criterios.append(accuracy_score(Y_dos, Y_Pronostico))
            context['criterios'] = criterios
            ReporteC =classification_report(Y_dos, Y_Pronostico, output_dict=True)
            RepClas = pd.DataFrame(ReporteC).transpose()
            context['ReporteClas'] = RepClas         

    #Dataframe de la importancia de variables
    Importancia = pd.DataFrame({'Variable': list(NuevaMat[predictoras]),
                            'Importancia': ModeloBA.feature_importances_}).sort_values('Importancia', ascending=False)
    context['Imp'] = Importancia

    #Reporte o arbol en texto
    Estimador = ModeloBA.estimators_[50]

    Reporte = export_text(Estimador, feature_names = predictoras)
    RepOut = []
    RepOut = Reporte.split("\n")
    context['reportes'] = RepOut

    #Eleccion para nuevo pronostico
    predictorasOut = NuevaMat[predictoras]
    context['Pred'] = predictorasOut.iloc[np.r_[0:5, -5:0]]

    return render(request, 'BosqueAleatorio2.html', context)

def BosqueAleatorioext2(request,pk, algType):
    context = {}
    context['pk'] = pk
    context['type'] = algType
    Val = request.POST.getlist('Nvals')
    X = pd.read_csv(os.path.join(BASE_DIR, 'IngeniousPrediction/data/info/X.csv'))
    Y = pd.read_csv(os.path.join(BASE_DIR, 'IngeniousPrediction/data/info/Y.csv'))

    #Division de los datos
    X_train, X_dos, Y_train, Y_dos = model_selection.train_test_split(X, Y, 
                                                                        test_size = 0.2, 
                                                                        random_state = 0, 
                                                                        shuffle = True)
    
    match algType:
        case 'P':
            context['AlgName'] = 'Pronostico'
            #Entrenamiento
            ModeloBA = RandomForestRegressor(random_state=0)
            flag = True
            
        case 'C':
            context['AlgName'] = 'Clasificacion'
            #Entrenamiento
            ModeloBA = RandomForestClassifier(random_state=0)
            flag = False

    ModeloBA.fit(X_train, Y_train)
    context['flag'] = flag

    col = list(X.columns)
    datoOut = {}
    print(X.shape[1])
    for i in range(X.shape[1]):
        datoOut[col[i]] = float(Val[i])
    Npron = pd.DataFrame(datoOut, index=[0])
    context['DfN'] = Npron
    resultado = ModeloBA.predict(Npron)
    print(resultado)
    context['resultado'] = resultado

    return render(request, 'BosqueAleatorio3.html', context)


def KMeans(request,pk):
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
    
