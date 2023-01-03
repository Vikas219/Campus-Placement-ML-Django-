from django.db import models

from django.contrib.auth.models import User

gender_choice = (
    ('male', 'male'),
    ('female', 'female')
)
# Create your models here.
class Predict(models.Model):
    name = models.CharField(max_length=120)
