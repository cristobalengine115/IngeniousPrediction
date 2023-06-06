from django.db import models

# Create your models here.

class Profesor (models.Model):
    name = models.CharField(max_length=200)
    Apellido = models.CharField(max_length=200)
    email = models.CharField(max_length=200)
    passwd = models.CharField(max_length=200)
