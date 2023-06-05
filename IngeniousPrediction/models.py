from django.db import models

# Create your models here.
class Profesor(models.Model):
    profe_id = models.BigAutoField(primary_key=True)
    nombre = models.CharField(max_length = 50, null=True)