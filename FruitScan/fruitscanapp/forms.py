from django import forms
from .models import UploadedImage
from django.contrib.auth.forms import UserCreationForm
from .models import CustomUser

class ImageForm(forms.ModelForm):

    class Meta:
        model = UploadedImage
        fields = ['image']


class RegistrationForm(UserCreationForm):
    class Meta:
        model = CustomUser
        fields = ['username', 'email', 'password1', 'password2']
