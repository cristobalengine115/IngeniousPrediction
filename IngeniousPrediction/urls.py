from django.urls import path, re_path, include
from . import views
urlpatterns =[
    path('', views.index),
    path('IP', views.GUI),
    path('validar', views.validar),
    path('registrar', views.registrar),
    path('registrarProyecto', views.registrarProyecto),

    path('edaSelector', views.edaSelector, name='edaSelector'),
    path('pcaSelector', views.pcaSelector, name='pcaSelector'),
    path('arbolSelector', views.arbolSelector, name='arbolSelector'),
    path('bosqueSelector', views.bosqueSelector, name='bosqueSelector'),
    path('segclaSelector', views.segclaSelector, name='segclaSelector'),

    path('IngeniousPrediction', views.index, name="IngeniousPrediction"),
    path('inicioGUI', views.inicioGUI, name="inicioGUI"),
    path('EDA/<int:pk>', views.EDA, name="EDA"),
    path('PCA/<int:pk>', views.PCA1, name='PCA'),
    path('PCA-2/<int:pk>', views.PCA2, name='PCA2'),
    path('ArbolDecision/<int:pk>/<str:algType>',views.ArbolDecision, name='ArbolDecision'),
    path('ArbolDecision_2/<int:pk>/<str:algType>',views.ArbolDecisionext, name='ArbolDecision_2'),
    path('ArbolDecision_3/<int:pk>/<str:algType>',views.ArbolDecisionext2, name='ArbolDecision_3'),
    
    path('BosqueAleatorio/<int:pk>/<str:algType>',views.BosqueAleatorio, name='BosqueAleatorio'),
    path('BosqueAleatorio_2/<int:pk>/<str:algType>',views.BosqueAleatorioext, name='BosqueAleatorio2'),
    path('BosqueAleatorio_3/<int:pk>/<str:algType>',views.BosqueAleatorioext2, name='BA3'),

    path('SegCla/<int:pk>',views.SegClas, name='SegmentacionClasificacion'),
    path('SegCla_2/<int:pk>',views.SegClas_2, name='SegmentacionClasificacion_2'),
    path('SegCla_3/<int:pk>',views.SegClas_3, name='SegmentacionClasificacion_3'),

    path('Proyectos/', views.lista_Proyectos, name='project_list'),

]