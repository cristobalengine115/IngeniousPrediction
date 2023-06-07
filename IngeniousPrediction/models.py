from django.db import models
class Proyecto(models.Model):
    nombre = models.CharField(max_length=80)
    descripcion = models.CharField(max_length=200)
    URL = models.CharField(max_length=150, null=True)
    data = models.FileField(upload_to='IngeniousPrediction/data/')

class Profesor(models.Model):
    nombre = models.CharField(max_length=200)
    apellido = models.CharField(max_length=200)
    email = models.CharField(max_length=200)
    passwd = models.CharField(max_length=200)