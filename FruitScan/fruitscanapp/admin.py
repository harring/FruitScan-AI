# Import necessary packages
from django.contrib import admin
from django.contrib.admin import AdminSite
from django.shortcuts import redirect
from .models import FruitClassification, MLWeights, ModelWeights, CustomUser, ImageData
from django.contrib.auth.admin import UserAdmin

class CustomAdminSite(AdminSite):
    def index(self, request, extra_context=None):
        """ Redirect user to customized admin page """
        return redirect('admin_view')

# Define admin site
admin_site = CustomAdminSite(name='customadmin')

# Register models to be used in the app for admin-related functionalities
admin.site.register(FruitClassification)
admin.site.register(MLWeights)
admin.site.register(ModelWeights)
admin.site.register(CustomUser, UserAdmin)
admin.site.register(ImageData)