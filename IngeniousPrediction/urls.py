from django.urls import path, re_path, include
from . import views
urlpatterns =[
    path('', views.index),
    path('validar', views.validar),
    path('registrar', views.registrar),
    path('registrarProyecto', views.registrarProyecto),
    path('IngeniousPrediction', views.index, name="IngeniousPrediction"),
    path('inicioGUI', views.inicioGUI, name="inicioGUI"),
    path('EDA/<int:pk>', views.EDA, name="EDA"),
    path('PCA', views.PCA, name="PCA"),
    path('ArbolDecision', views.ArbolDecision, name="ArbolDecision"),
    path('BosqueAleatorio', views.BosqueAleatorio, name="BosqueAleatorio"),
    path('K-Means', views.KMeans, name='K-Means'),
    path('creaProyecto/', views.crea_Proyecto, name='upload_project'),
    path('Proyectos/', views.lista_Proyectos, name='project_list'),

]