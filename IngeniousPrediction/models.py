from django.db import models
from django.core import validators




class Profesor(models.Model):
    nombre = models.CharField(max_length=200)
    apellido = models.CharField(max_length=200)
    email = models.CharField(max_length=200)
    passwd = models.CharField(max_length=200)

class Proyecto(models.Model):
    name = models.CharField(max_length=80)
    description = models.CharField(max_length=200)
    URL = models.CharField(max_length=150, null=True)
    data = models.FileField(upload_to='IngeniousPrediction/data/')

    def __str__(self):
        return self.title

    def delete(self, *args, **kwargs):
        self.data.delete()
        super().delete(*args, **kwargs)
