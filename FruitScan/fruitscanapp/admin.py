from django.contrib import admin
from django.contrib.admin import AdminSite
from django.urls import path
from django.shortcuts import redirect
from .models import FruitClassification, MLWeights, ModelWeights, CustomUser, ImageData
from django.contrib.auth.admin import UserAdmin

class CustomAdminSite(AdminSite):
    def index(self, request, extra_context=None):
        return redirect('admin_view')

admin_site = CustomAdminSite(name='customadmin')

admin.site.register(FruitClassification)
admin.site.register(MLWeights)
admin.site.register(ModelWeights)
admin.site.register(CustomUser, UserAdmin)
admin.site.register(ImageData)