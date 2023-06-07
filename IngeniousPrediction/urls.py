from django.urls import path, re_path, include
from . import views
urlpatterns =[
    path('', views.index),
    path('IP', views.GUI),
    path('validar', views.validar),
    path('registrar', views.registrar),
    path('registrarProyecto', views.registrarProyecto),
    path('IngeniousPrediction', views.index, name="IngeniousPrediction"),
    path('inicioGUI', views.inicioGUI, name="inicioGUI"),
    path('EDA/<int:pk>', views.EDA, name="EDA"),
    path('PCA/<int:pk>', views.PCA1, name='PCA'),
    path('PCA_2/<int:pk>', views.PCA2, name='PCA2'),
    path('ArbolDecision/<int:pk>/<str:algType>',views.ArbolDecision, name='ArbolDecision'),
    path('ArbolDecision_2/<int:pk>/<str:algType>',views.ArbolDecisionext, name='ArbolDecision_2'),
    path('ArbolDecision_3/<int:pk>/<str:algType>',views.ArbolDecisionext2, name='ArbolDecision_3'),
    path('BosqueAleatorio/<int:pk>/<str:algType>',views.BosqueAleatorio, name='BosqueAleatorio'),
    path('BosqueAleatorio_2/<int:pk>/<str:algType>',views.BosqueAleatorioext, name='BosqueAleatorio2'),
    path('BosqueAleatorio_3/<int:pk>/<str:algType>',views.BosqueAleatorioext2, name='BA3'),
    path('SegCla/<int:pk>',views.SegClas, name='SegmentacionClasificacion'),
    path('SegCla_2/<int:pk>',views.SegClas_2, name='SegmentacionClasificacion_2'),
    path('SegCla_3/<int:pk>',views.SegClas_3, name='SegmentacionClasificacion_3'),
    path('SVM/<int:pk>/<str:algType>', views.SVM, name='SVM'),
    path('SVM_2/<int:pk>/<str:algType>', views.SVM_2, name='SVM2'),
    path('SVM_3/<int:pk>/<str:algType>', views.SVM_3, name='SVM3'),

    path('K-Means', views.KMeans, name='K-Means'),
    path('creaProyecto/', views.crea_Proyecto, name='upload_project'),
    path('Proyectos/', views.lista_Proyectos, name='project_list'),

]