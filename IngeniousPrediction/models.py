from django.db import models
from django.core import validators



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