from django.db import models

# Create your models here.

class Profesor(models.Model):
    nombre = models.CharField(max_length=200)
    apellido = models.CharField(max_length=200)
    email = models.CharField(max_length=200)
    passwd = models.CharField(max_length=200)

class Grupo(models.Model):
    descripcion = models.CharField(max_length=500)
    cantidad = models.CharField(max_length=500)