from django.urls import path, re_path, include
from . import views
urlpatterns =[
    path('', views.index),
    path('validar', views.validar),
    path('registrar', views.registrar),
    path('IngeniousPrediction', views.index, name="IngeniousPrediction"),
    path('EDA', views.EDA, name="EDA"),
    path('PCA', views.PCA, name="PCA"),
    path('ArbolDecision', views.ArbolDecision, name="ArbolDecision"),
    path('BosqueAleatorio', views.BosqueAleatorio, name="BosqueAleatorio"),
    path('K-Means', views.KMeans, name='K-Means')

]