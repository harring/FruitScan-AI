# Contributors: Mijin, Jonathan, Patricia, Erik

# Import necessary packages
from django.db import models
from django.db.models import JSONField
from django.contrib.auth.models import AbstractUser

class FruitClassification(models.Model):
    """ Create model (schema) for image input """
    # True label for fruit in numeric values
    label = models.IntegerField()
    # This field can store JSON-serializable data, which includes lists and dictionaries.
    pixels = JSONField()
    def __str__(self):
        return f"{self.label}"

# Model for ML model weights
class MLWeights(models.Model):
    """ Create model (schema) for ML model weights """
    # Specify version field
    version = models.CharField(max_length=5)
    # Weights to be used in classification
    # This can be changed depending on our final model.
    # If the final model have many neurons, thus many weights, perhaps it's better to save them
    # in an array and keep the info in JSONField. 
    weight = JSONField()
    # Otherwise we can create separate columns for each weight
    weight_0 = models.DecimalField(max_digits=10, decimal_places=2)
    weight_1 = models.DecimalField(max_digits=10, decimal_places=2)
    weight_2 = models.DecimalField(max_digits=10, decimal_places=2)

class UploadedImage(models.Model):
    """ Model for the uploaded pictures, specifying the uploaded path """
    image = models.ImageField(upload_to='images/')

class ModelWeights(models.Model):
    """ Create model (schema) for model weights along with confusion matrix, accuracy metrics """
    version = models.CharField(max_length=5)
    path = models.FileField(upload_to='ModelWeights/')
    confusion_matrix = models.ImageField(upload_to='Performance/ConfusionMatrix/')
    train_accuracy = models.DecimalField(max_digits=10, decimal_places=3, default=0)
    val_accuracy = models.DecimalField(max_digits=10, decimal_places=3, default=0)
    test_set_accuracy = models.DecimalField(max_digits=10, decimal_places=3, default=0)

class ImageData(models.Model):
    """ Create model (schema) for image data"""
    label = models.CharField(max_length=1)
    image_data = models.BinaryField()

class TestImageData(models.Model):
    """ Create model (schema) for test image data """
    label = models.CharField(max_length=1)
    image_data = models.BinaryField()

class CustomUser(AbstractUser):
    """ Create model (schema) for registered users and their search history """
    pred_hist_img = models.ImageField(upload_to='images', blank=True)
    def __str__(self):
        return self.username

def user_directory_path(instance, filename):
    """ This function generates the file path for uploaded images """
    return f'images/user_{instance.user.id}/{filename}'

class UserImage(models.Model):
    """ Create model (schema) for user image with connection between images and user"""
    user = models.ForeignKey(CustomUser, related_name='user_images', on_delete=models.CASCADE)
    image = models.ImageField(upload_to=user_directory_path, blank=True)
    pred = models.CharField(max_length=50)