from django.urls import path, re_path
from . import views
urlpatterns =[
    path('', views.index),
    path('IP', views.GUI),
    path('EDA', views.AlgoritmoEDA)
]