# Import necessary packages
from django import forms
from .models import UploadedImage
from django.contrib.auth.forms import UserCreationForm
from .models import CustomUser

class ImageForm(forms.ModelForm):
    """ Define a form to take an image from user """
    class Meta:
        model = UploadedImage
        fields = ['image']


class RegistrationForm(UserCreationForm):
    """ Define a form to take user registation information """
    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'password1', 'password2']
