from django.db import models

class Proyecto(models.Model):
    Nombre = models.CharField(max_length=80)
    descripcion = models.CharField(max_length=200)
    URL = models.CharField(max_length=150, null=True)
    data = models.FileField(upload_to='IngeniousPrediction/data/')

<<<<<<< HEAD
class Profesor(models.Model):
    nombre = models.CharField(max_length=200)
    apellido = models.CharField(max_length=200)
    email = models.CharField(max_length=200)
    passwd = models.CharField(max_length=200)

class Grupo(models.Model):
    descripcion = models.CharField(max_length=500)
    cantidad = models.CharField(max_length=500)
=======
    def __str__(self):
        return self.title

    def delete(self, *args, **kwargs):
        self.data.delete()
        super().delete(*args, **kwargs)

class Profesor(models.Model):
    profe_id = models.BigAutoField(primary_key=True)
    nombre = models.CharField(max_length = 50, null=True)
>>>>>>> test
